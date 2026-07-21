import torch
import triton
import triton.language as tl

@triton.jit
def awq_gemm_kernel(
    x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, y_ptr,
    M, N, K,
    group_size: tl.constexpr, pack_factor: tl.constexpr,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,     # 建议设置为 group_size 整数倍
):
    G = K // group_size
    BLOCK_G = BLOCK_K // group_size
    
    PACK_N = N // pack_factor
    PACK_BLOCK_N = BLOCK_N // pack_factor
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rpackn = pid_n * PACK_BLOCK_N + tl.arange(0, PACK_BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rg = tl.arange(0, BLOCK_K // group_size)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    qzeros_ptrs = qzeros_ptr + rg[:, None] * stride_qzg + rpackn[None, :] * stride_qzn 
    qweight_ptrs = qweight_ptr + rk[:, None] * stride_qwk + rpackn[None, :] * stride_qwn
    scales_ptrs = scales_ptr + rg[:, None] * stride_sg + rn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        g = k // group_size
        
        # ---- 1. 读 x tile ----
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # ---- 2. 读 qweight tile 并解包成 int4 ----
        qzeros_mask = (rg[:, None] + g < G) & (rpackn[:, None] < PACK_N)
        qweight_mask = (rk[:, None] + k < K) & (rpackn[:, None] < PACK_N)
        qzeros_tile = tl.load(qzeros_ptrs, mask=qzeros_mask, other=0.0)
        qweight_tile = tl.load(qweight_ptrs, mask=qweight_mask, other=0.0)
        
        shifts = tl.arange(0, 8) * 4
        # 按高低位拆分成nibbles
        qweight_nibbles = (qweight_tile[:, None] >> shifts[None, :]) & 0x0F
        qzeros_nibbles = (qzeros_tile[:, None] >> shifts[None, :]) & 0x0F
        
        unpack_weight = qweight_nibbles.reshape(BLOCK_K, BLOCK_N, can_reorder=False)
        unpack_zeros = qzeros_nibbles.reshape(BLOCK_G, BLOCK_N, can_reorder=False)

        # ---- 3. 读 scale / zero（当前 K-tile 对应第几个 group？） ----
        scales_mask = (rg[:, None] + g < G) & (rn[None, :] < N)
        scales_tile = tl.load(scales_ptrs, mask=scales_mask, other=0.0)
        
        # 将每一行交错复制 group_size 次
        zeros_expanded = tl.repeat_interleave(unpack_zeros, group_size, dim=0) - 1 # 量化过程中会对零点+1
        scales_expanded = tl.repeat_interleave(scales_tile, group_size, dim=0)
        
        # ---- 4. 反量化成 w_fp16 ----
        w_fp16 = (unpack_weight - zeros_expanded) * scales_expanded
        w_fp16 = w_fp16.to(tl.float16)

        # ---- 5. 累加 ----
        accumulator += tl.dot(x_tile, w_fp16)

        # ---- 6. 指针平移 ----
        x_ptrs += BLOCK_K * stride_xk
        qweight_ptrs += BLOCK_K * stride_qwk
        qzeros_ptrs += BLOCK_G * stride_qzg
        scales_ptrs += BLOCK_G * stride_sg

    y = accumulator.to(tl.float16)
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    M, K = x.shape
    K2, N_packed = qweight.shape
    N = N_packed * pack_factor
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, group_size   # 先用小 tile 保证正确性，后面再调优
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    awq_gemm_kernel[grid](
        x, qweight, qzeros, scales, y,
        M, N, K,
        group_size, pack_factor,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    if bias is not None:
        y += bias
    return y