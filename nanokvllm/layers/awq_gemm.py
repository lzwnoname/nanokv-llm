import torch
from torch import nn
import torch.nn.functional as F

class AWQGemmTorch():
  # 用F.linear + dequantization实现，保证正确性
  def __call__(self, x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    w_fp16 = dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor)
    # w_fp_16 -> [in, out] 需要转置
    return F.linear(x, w_fp16.t().contiguous(), bias)

class AWQGemmTriton():
  # 用手写Triton算子实现，提高速度
  def __call__(self, x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    from nanokvllm.kernel.awq_triton_kernel import awq_gemm_kernel_launch
    return awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor, bias)


def get_awq_gemm(kernel: str):
  if kernel == "torch":
    return AWQGemmTorch()
  elif kernel == "triton":
    return AWQGemmTriton()
  else:
    raise NotImplementedError

def dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor):
  # AutoAWQ GEMM 4bit 打包顺序（见 awq/modules/linear/gemm.py 的 pack）：
  #   int32 中第 i 个 nibble (shift i*w_bit) 存的是 output col order_map[i] = [0,2,4,6,1,3,5,7][i]。
  # 其逆序 inv_order = [0,4,1,5,2,6,3,7]（即 3-bit 循环右移）。用 inv_order 作为 shift 顺序，
  # 解出的 nibble 即按自然 output 列序排列，无需额外 permute。
  w_bit = 32 // pack_factor
  # inv_order=[0,4,1,5,2,6,3,7] 用 on-device 算术构造（3-bit 循环右移），
  # 避免从 Python list 建 CUDA tensor 触发 CPU→CUDA 拷贝（会破坏 CUDA graph 捕获）。
  p = torch.arange(pack_factor, dtype=torch.int32, device=qweight.device)
  inv_order = ((p & 1) << 2) | (((p >> 2) & 1) << 1) | ((p >> 1) & 1)
  shifts = inv_order * w_bit  # [0,16,4,20,8,24,12,28]

  in_features, out_packed = qweight.shape
  out_features = out_packed * pack_factor

  # qweight [in, out//pack] -> nibbles [in, out//pack, pack] (按 inv_order 取，已是自然列序) -> [in, out]
  int4_vals = ((qweight[:, :, None].to(torch.int32) >> shifts[None, None, :]) & 0x0F) \
      .reshape(in_features, out_features)

  # qzeros [in//group, out//pack] -> [in//group, out]
  int4_zeros = ((qzeros[:, :, None].to(torch.int32) >> shifts[None, None, :]) & 0x0F) \
      .reshape(in_features // group_size, out_features)

  # 注意：AutoAWQ 0.2.x 的 qzeros 直接存的是 real zero_point（量化时未做 +1 偏移，
  # 见 awq/modules/linear/gemm.py 的 pack 与 awq/quantize/quantizer.py 的 zeros 计算），
  # 因此反量化直接用 (int4 - zero) * scale，不要再做 -1。

  # 将scale和qzeros广播成 -> [in, out]
  scales_expanded = scales.repeat_interleave(group_size, dim=0)
  zeros_expanded = int4_zeros.repeat_interleave(group_size, dim=0)

  # 反量化用 float32 累加保证精度，输出 dtype 对齐 scales（即模型 dtype，bf16 或 fp16），
  # 避免与 hidden_states 的 dtype 冲突。
  w = (int4_vals.float() - zeros_expanded.float()) * scales_expanded.float()
  return w.to(scales.dtype)
  