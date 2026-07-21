# AWQ 反量化（dequantize）实现方案

> 目标：实现 `dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor) -> w_fp16`
> 要求：代码优美、可读、尽量向量化（避免 Python for 循环遍历 group）、正确性可验证。
> 本文只给方案和伪代码，不改现有代码。

---

## 1. 输入输出规格

```
输入：
  qweight: [in_features, out_features // pack_factor]   int32
           每个 int32 打包了 pack_factor 个 int4 值（AWQ pack_factor=8）
  qzeros:  [in_features // group_size, out_features // pack_factor]  int32
           每个 int32 同样打包 pack_factor 个 int4 zero_point
  scales:  [in_features // group_size, out_features]   float16
           每个 group、每个输出通道一个 scale（未打包）
  group_size: int（通常 128）
  pack_factor: int（4bit → 8）

输出：
  w_fp16:  [in_features, out_features]   float16
           反量化后的权重，可直接 .t() 后喂给 F.linear
```

---

## 2. 反量化数学公式

对每个权重元素 `w`：

```
# 量化时的编码：
q = clip(round(w / scale) + zero_point, 0, 2^bits - 1)    # q ∈ [0, 15]（4bit）

# 反量化解码：
w ≈ (q - zero_point) * scale
```

其中 `scale` 和 `zero_point` 是按 group 共享的（同一个 group 内所有权重用同一对 scale/zero_point）。

---

## 3. AutoAWQ 打包格式细节（必须对齐）

### 3.1 qweight 的打包顺序

AutoAWQ 不是简单地把 8 个 int4 顺序拼进 int32，而是用了一个**交织顺序**：

```
原始 int4 序列：  [q0, q1, q2, q3, q4, q5, q6, q7]
打包进 int32 后的位分布（从低位到高位）：
  q0 q1 q2 q3 q4 q5 q6 q7   （实际上是 [0,1,2,3,4,5,6,7] 顺序，每 4 bit 一个）

但 AutoAWQ CUDA kernel 期望的解包顺序是：
  bit[0:4]   → q0
  bit[4:8]   → q1
  ...
  bit[28:32] → q7
```

**关键**：需要确认 AutoAWQ 实际打包用的是**顺序拼接**还是**交织拼接**。查看 AutoAWQ 源码 `awq/quantize/quantizer.py::pack_int4` 确认。常见的 AutoAWQ GEMM 格式是**顺序拼接**（不是 GPTQ 的交织），即：

```python
# AutoAWQ pack 伪代码
for i in range(pack_factor):
    qword |= (int4_values[i] << (4 * i))
```

### 3.2 qzeros 的偏移

AutoAWQ 的 qzeros 存储时做了 **+1 偏移**（防止 zero_point=0 与"未初始化"混淆）：

```
存储的 qzeros 值 = 实际 zero_point + 1
反量化时：zero_point = stored_qzeros - 1
```

**必须确认**：不同 AutoAWQ 版本可能不一致，需要用一个小模型量化后做数值对比验证。

---

## 4. 优美实现方案：纯 PyTorch 向量化（无 for 循环）

### 4.1 核心思路

把"解包 int32 → 8 个 int4"这一步用**位运算 + reshape** 完成，不写 Python for 循环。

```
qweight: [in, out//pack]  int32

Step 1: view as uint8（每个 int32 → 4 个 uint8，每个 uint8 含 2 个 int4）
  qweight.view(torch.uint8)  →  [in, out//pack, 4]  uint8

Step 2: 拆成高低位
  low_nibbles  = uint8 & 0x0F        → 低 4 bit
  high_nibbles = (uint8 >> 4) & 0x0F → 高 4 bit

Step 3: 交织回正确顺序
  int4_values = stack([low, high], dim=-1).reshape(in, out//pack, 8)
  → [in, out//pack * 8] = [in, out]  每个 int4 ∈ [0, 15]

Step 4: 广播 scales 和 zeros
  scales: [in//group, out] → 广播到 [in, out]
  zeros:  [in//group, out//pack] → 解包成 [in//group, out] → 广播到 [in, out]

Step 5: 反量化
  w_fp16 = (int4_values - zeros) * scales
```

### 4.2 伪代码

```python
def dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor):
    """
    纯 PyTorch 向量化反量化。
    输入均为 AutoAWQ GEMM 格式。
    返回 [in, out] float16。
    """
    in_features, out_packed = qweight.shape
    out_features = out_packed * pack_factor

    # ===== Step 1: 解包 qweight → [in, out] int32 (值域 [0, 15]) =====
    # view as uint8: [in, out//pack, 4]
    qw_bytes = qweight.view(torch.uint8).reshape(in_features, out_packed, 4)

    # 拆高低 nibble
    low  = (qw_bytes & 0x0F).to(torch.int32)          # [in, out//pack, 4]
    high = ((qw_bytes >> 4) & 0x0F).to(torch.int32)   # [in, out//pack, 4]

    # 交织：[q0,q1,q2,...,q7] 对应 [byte0_low, byte0_high, byte1_low, byte1_high, ...]
    int4_vals = torch.stack([low, high], dim=-1).reshape(in_features, out_features)
    # int4_vals: [in, out], 值域 [0, 15]

    # ===== Step 2: 解包 qzeros → [in//group, out] int32 =====
    num_groups = in_features // group_size
    qz_bytes = qzeros.view(torch.uint8).reshape(num_groups, out_packed, 4)
    qz_low  = (qz_bytes & 0x0F).to(torch.int32)
    qz_high = ((qz_bytes >> 4) & 0x0F).to(torch.int32)
    zeros_int4 = torch.stack([qz_low, qz_high], dim=-1).reshape(num_groups, out_features)
    # AutoAWQ 存储时 +1 偏移，反量化时减回
    zeros_int4 = zeros_int4 - 1    # 值域 [-1, 14]

    # ===== Step 3: 广播到 [in, out] =====
    # scales: [num_groups, out] → [in, out]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    # zeros:  [num_groups, out] → [in, out]
    zeros_expanded = zeros_int4.repeat_interleave(group_size, dim=0)

    # ===== Step 4: 反量化 =====
    w_fp16 = (int4_vals.float() - zeros_expanded.float()) * scales_expanded.float()
    return w_fp16.to(torch.float16)
```

### 4.3 为什么这个方案"优美"

| 特性 | 说明 |
|---|---|
| **零 Python for 循环** | 全部用 `view`/`reshape`/`stack`/`repeat_interleave`，GPU 友好 |
| **可读性强** | 5 个 step 逐行对应数学公式，从打包格式到反量化一气呵成 |
| **正确性易验证** | 跟 AutoAWQ 官方 `dequantize` 做数值对比即可 |
| **无额外依赖** | 纯 `torch`，不需要 Triton/CUDA 扩展 |
| **shape 清晰** | 每步都标注了 shape，方便调试 |

---

## 5. 注意事项与验证

### 5.1 打包顺序必须验证

上述伪代码假设 AutoAWQ 用的是 **"byte 内低 nibble 在前"** 的顺序（`[low0, high0, low1, high1, ...]`）。实际可能因版本不同而异。

**验证方法**：

```python
# 1. 用 AutoAWQ 量化一个小矩阵
import torch
from awq.quantize.quantizer import pseudo_quantize_tensor, pack_int4

w = torch.randn(128, 256, dtype=torch.float16)
# ... 量化打包 ...

# 2. 用自己的 dequantize 反量化
w_dq = dequantize_awq_weight(qweight, qzeros, scales, 128, 8)

# 3. 对比 AutoAWQ 官方反量化
w_dq_official = autoawq_dequantize(qweight, qzeros, scales, 128, 8)

assert torch.allclose(w_dq, w_dq_official, atol=1e-3)
```

如果不一致，调整 Step 1 的 `stack` 顺序（如 `[high, low]` 而非 `[low, high]`）。

### 5.2 qzeros +1 偏移

AutoAWQ 较新版本（0.2+）存储 qzeros 时做了 +1 偏移。如果量化配置 `zero_point=False`（对称量化），则 qzeros 全为 0，反量化时不需要减。需要根据 `quantize_config.json` 的 `zero_point` 字段判断。

### 5.3 group_size 对齐

`in_features` 必须能被 `group_size` 整除，否则 `repeat_interleave` 会报错。在 `Config.__post_init__` 里加断言。

### 5.4 内存开销

反量化后 `w_fp16: [in, out]` 是临时 tensor，每次 forward 都会重新分配。对于 8B 模型单层约 `[4096, 4096]` = 32MB，可以接受。但如果用 CUDA Graph，需要确认这个临时分配不破坏 graph 捕获（可能需要预分配 buffer）。

---

## 6. 性能说明（Phase A 的定位）

这个实现是 **Phase A（正确性优先）**，定位是"基线/验证用"，**不是性能优化版本**：

- 每次调用都会反量化**整层权重**到 FP16，没有省显存带宽
- 比 FP16 推理更慢（多了一次反量化开销）
- 价值：验证 qweight/qzeros/scales 加载正确、格式对齐、数学等价

真正的性能收益在 Phase B（Triton fused kernel），那时反量化在 kernel 内部逐 tile 进行，不写回完整 FP16 权重。

---

## 7. 完整调用链路

```
AWQLinear.forward(x)
  → self.awq_gemm(x, qweight, qzeros, scales, group_size, pack_factor, bias)
    → AWQGemmTorch.__call__()
      → dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor)
          → w_fp16: [in, out] float16
      → F.linear(x, w_fp16.t().contiguous(), bias)
          → y: [batch, out]
```
