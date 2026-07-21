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
  in_features, out_packed = qweight.shape
  out_features = out_packed * pack_factor
  
  # 解包qweight -> [in, out * 4]
  qw_bytes = qweight.view(torch.uint8).reshape(in_features, out_packed, 4)
  
  # 拆解高低位
  low_value = (qw_bytes & 0x0F).to(torch.int32)
  high_value = ((qw_bytes >> 4) & 0x0F).to(torch.int32)
  
  # 交织排布 q1_low, q1_high, q2_low, q2_high...
  int4_vals = torch.cat([low_value, high_value], dim=-1).reshape(in_features, out_features)
  
  # 解包qzeros -> [in // group_size, out * 4]
  qz_bytes = qzeros.view(torch.uint8).reshape(in_features // group_size, out_packed, 4)
  low_value = (qz_bytes & 0x0F).to(torch.int32)
  high_value = ((qz_bytes >> 4) & 0x0F).to(torch.int32)
  int4_zeros = torch.cat([low_value, high_value], dim=-1).reshape(in_features // group_size, out_features)
  
  # AutoAWQ在量化时会将zero_point偏移+1，反量化要剪回来
  int4_zeros -= 1
  
  # 将scale和qzeros广播成 -> [in, out * 4]
  scales_expanded = scales.repeat_interleave(group_size, dim=0)
  zeros_expanded = int4_zeros.repeat_interleave(group_size, dim=0)
  
  w_fp16 = (int4_vals.float() - zeros_expanded.float()) * scales_expanded.float()
  return w_fp16.to(torch.float16)
  