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
    PACK_BLOCK_N: tl.constexpr,  # = BLOCK_N // pack_factor，须作为 constexpr 显式传入
    BLOCK_G: tl.constexpr,       # = BLOCK_K // group_size，须作为 constexpr 显式传入
):
    G = K // group_size

    PACK_N = N // pack_factor

    # 输出/中间反量化 dtype 对齐 scales（即模型 dtype，bf16 或 fp16）。
    # scales_ptr.dtype.element_ty 是编译期常量，可以在循环外和 store 处安全引用。
    out_dtype = scales_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rpackn = pid_n * PACK_BLOCK_N + tl.arange(0, PACK_BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rg = tl.arange(0, BLOCK_G)

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
        # mask 形状须为 [K_dim, N_packed_dim]：K 维用 [:, None]，N_packed 维用 [None, :]
        qzeros_mask = (rg[:, None] + g < G) & (rpackn[None, :] < PACK_N)
        qweight_mask = (rk[:, None] + k < K) & (rpackn[None, :] < PACK_N)
        qzeros_tile = tl.load(qzeros_ptrs, mask=qzeros_mask, other=0.0)
        qweight_tile = tl.load(qweight_ptrs, mask=qweight_mask, other=0.0)
        
        # AutoAWQ GEMM 4bit pack 顺序 order_map=[0,2,4,6,1,3,5,7]（见 awq/modules/linear/gemm.py）。
        # 其逆 inv_order=[0,4,1,5,2,6,3,7]（3-bit 循环右移）。用 inv_order 作为 shift 顺序，
        # 解出的 nibble 即按自然 output 列序排列，无需额外 permute。
        # inv_order[p] = 3-bit rotate-right(p) = (bit0<<2)|(bit2<<1)|bit1
        p = tl.arange(0, 8)
        inv_order = ((p & 1) << 2) | (((p >> 2) & 1) << 1) | ((p >> 1) & 1)
        shifts = inv_order * (32 // pack_factor)
        # 显式构造 rank-3 广播，避免 [:, None] 在 2D 上插入中间维的歧义
        # qweight_tile [BLOCK_K, PACK_BLOCK_N] -> [:, :, None] = [BLOCK_K, PACK_BLOCK_N, 1]
        # shifts [8] -> [None, None, :] = [1, 1, 8]；结果 [BLOCK_K, PACK_BLOCK_N, 8]（已是自然列序）
        qweight_nibbles = (qweight_tile[:, :, None] >> shifts[None, None, :]) & 0x0F
        qzeros_nibbles = (qzeros_tile[:, :, None] >> shifts[None, None, :]) & 0x0F
        
        unpack_weight = qweight_nibbles.reshape(BLOCK_K, BLOCK_N, can_reorder=False)
        unpack_zeros = qzeros_nibbles.reshape(BLOCK_G, BLOCK_N, can_reorder=False)

        # ---- 3. 读 scale / zero（当前 K-tile 对应第几个 group？） ----
        scales_mask = (rg[:, None] + g < G) & (rn[None, :] < N)
        scales_tile = tl.load(scales_ptrs, mask=scales_mask, other=0.0)
        
        # 将每一行交错复制 group_size 次
        # tl.repeat_interleave 在 triton 3.6 不存在，用 broadcast_to+reshape 等价实现：
        # [BLOCK_G, BLOCK_N] -> [:, None, :] = [BLOCK_G, 1, BLOCK_N] -> broadcast [BLOCK_G, group_size, BLOCK_N]
        # -> reshape [BLOCK_G*group_size, BLOCK_N] = [BLOCK_K, BLOCK_N]，即每 group 行重复 group_size 次
        # 注意：AutoAWQ 0.2.x 的 qzeros 直接存 real zero_point（无 +1 偏移），这里不再 -1
        zeros_expanded = tl.broadcast_to(
            unpack_zeros[:, None, :], (BLOCK_G, group_size, BLOCK_N)
        ).reshape(BLOCK_K, BLOCK_N, can_reorder=False)
        scales_expanded = tl.broadcast_to(
            scales_tile[:, None, :], (BLOCK_G, group_size, BLOCK_N)
        ).reshape(BLOCK_K, BLOCK_N, can_reorder=False)
        
        # ---- 4. 反量化成 fp16/bf16（对齐 scales.dtype，即模型 dtype） ----
        w = (unpack_weight - zeros_expanded) * scales_expanded
        w = w.to(out_dtype)

        # ---- 5. 累加 ----
        accumulator += tl.dot(x_tile, w)

        # ---- 6. 指针平移 ----
        x_ptrs += BLOCK_K * stride_xk
        qweight_ptrs += BLOCK_K * stride_qwk
        qzeros_ptrs += BLOCK_G * stride_qzg
        scales_ptrs += BLOCK_G * stride_sg

    y = accumulator.to(out_dtype)
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    M, K = x.shape
    K2, N_packed = qweight.shape
    N = N_packed * pack_factor
    assert K == K2

    # 输出 dtype 对齐 scales（即模型 dtype，bf16 或 fp16），避免与 hidden_states 冲突
    y = torch.empty((M, N), device=x.device, dtype=scales.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, group_size   # 先用小 tile 保证正确性，后面再调优
    PACK_BLOCK_N = BLOCK_N // pack_factor
    BLOCK_G = BLOCK_K // group_size
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
        PACK_BLOCK_N=PACK_BLOCK_N, BLOCK_G=BLOCK_G,
    )
    if bias is not None:
        y += bias
    return y