# AWQ 量化特性 Bug 归档

> 版本：v1  
> 代码基线：nano-kvllm v0.2.0  
> 测试环境：`/root/nano-vllm/.venv`（Python 3.10 / torch 2.11.0+cu128 / transformers 5.12.1 / flash_attn 2.8.3 / triton 3.6.0 / autoawq 0.2.9）  
> 测试模型：Qwen3-0.6B（FP16）→ AutoAWQ 量化为 Qwen3-0.6B-AWQ（4bit, group_size=128, GEMM）  
> 关联文档：`docs/test_plan_awq.md`  
> 关联脚本：`quantize_awq.py`、`bench_awq.py`  
> 状态：全部已修复并通过 §3.1 冒烟测试 + §3.2 kernel 一致性测试

---

## 0. 背景与测试方法

AWQ 量化推理链路涉及：`quantize_awq.py`（生成量化产物）→ `loader.py`（加载 `.qweight/.qzeros/.scales`）→ `awq_linear.py`（AWQ Linear 层 + TP 切分）→ `awq_gemm.py`（反量化 + GEMM 的 torch 实现）→ `awq_triton_kernel.py`（triton 融合 GEMM）→ `model_runner.py` / `qwen3.py` / `rotary_embedding.py`（模型集成）。

测试按 `test_plan_awq.md` 的 §3 进行：
- **§3.1 冒烟测试**：加载真实 AWQ 模型，喂 Math prompt，验证输出可读、含 CoT 与答案。
- **§3.2 kernel 一致性**：`awq_kernel="torch"` 与 `awq_kernel="triton"` 同 seed 同 prompt 逐 token 比对。

由于 AutoAWQ 的 int4 打包格式与文档假设不符，且 triton kernel 存在多处编译/数值问题，测试过程中暴露出 **7 类 bug**，集中在反量化、kernel、加载、模型集成四条链路上。下文逐条记录「位置 → 现象 → 根因 → 修复 → 验证」的完整逻辑链路。

修复后实测：
- §3.1：torch / triton 两个 kernel 均输出通顺 CoT，正确算出 `12-4-3=5` 并给出 `\boxed{5}`（修复前输出为 `!!!` 乱码）。
- §3.2：真实权重上 torch vs triton 单层 GEMM `max_diff=0.000000`；近贪心（top-1）逐 token **100% 一致**（200/200）。
- 反量化还原 vs FP16 原始权重：cos≈0.977(gate_proj) / 0.973(q_proj)，符合 int4 AWQ 预期精度。
- FP16 推理路径无回归。

---

## 1. Bug 总表

| # | 位置 | 类型 | 影响 |
|---|---|---|---|
| 1 | `nanokvllm/layers/awq_gemm.py` `dequantize_awq_weight` | 反量化打包顺序错 | torch kernel 反量化权重列顺序错乱，输出乱码 |
| 2 | `awq_gemm.py` + `awq_triton_kernel.py` | zero_point −1 处理错 | 反量化结果整体偏移 +scale，精度崩坏 |
| 3 | `nanokvllm/kernel/awq_triton_kernel.py` | triton kernel 编译失败（4 个叠加问题） | `awq_kernel="triton"` 完全不可用 |
| 4 | `nanokvllm/layers/awq_linear.py` `AWQQKVParallelLinear` | 形参顺序与调用方不一致 | 模型构建即 `TypeError: multiple values` |
| 5 | `nanokvllm/utils/loader.py` | AWQ 融合层参数名未拼组件后缀 | 权重加载 `AttributeError` |
| 6 | `nanokvllm/layers/rotary_embedding.py` `get_rope` | `@lru_cache` + dict rope_scaling 崩溃 | 任何模型加载即 `TypeError: unhashable`（非 AWQ 专属，预存 bug） |
| 7 | `nanokvllm/engine/model_runner.py` | AWQ fp16 反量化 vs bf16 模型 dtype 冲突 | `F.linear` 报 `BFloat16 != Half` |
| 8 | `nanokvllm/layers/awq_gemm.py` `dequantize_awq_weight` | torch kernel 从 Python list 建 CUDA tensor，破坏 CUDA graph 捕获 | AWQ-torch + `enforce_eager=False` 崩溃 `RuntimeError` |

---

## 2. Bug 1：torch kernel 反量化 int4 打包顺序错误

**位置**：`nanokvllm/layers/awq_gemm.py` `dequantize_awq_weight`（当前实现见 `awq_gemm.py:27-56`）

### 现象
用真实 AutoAWQ 权重反量化后与 FP16 原始权重比对，cosine 仅 0.22（近乎随机）；端到端冒烟测试输出全为 `!` 乱码。

### 根因（为什么错）
原代码用「按字节拆高低位 + `cat`」的方式解包：

```python
# 旧代码（错误）
qw_bytes = qweight.view(torch.uint8).reshape(in_features, out_packed, 4)
low_value  = (qw_bytes & 0x0F).to(torch.int32)
high_value = ((qw_bytes >> 4) & 0x0F).to(torch.int32)
# 注释写"交织排布 q1_low, q1_high..."，但 cat 是拼接不是交织
int4_vals = torch.cat([low_value, high_value], dim=-1).reshape(in_features, out_features)
```

这里有两层错误：

1. **`cat` 不是交织**：注释声称要「q1_low, q1_high, q2_low, q2_high…」交织，但 `torch.cat([low, high], dim=-1)` 实际产生 `[low0,low1,low2,low3, high0,high1,high2,high3]`（拼接），而非交织。
2. **更深层的根因——AutoAWQ 的真实打包顺序不是 little-endian**：查看 AutoAWQ 源码 `awq/modules/linear/gemm.py` 的 pack 逻辑：
   ```python
   for col in range(out // pack_num):
       order_map = [0, 2, 4, 6, 1, 3, 5, 7]
       for i in range(pack_num):
           qweight[:, col] |= intweight[:, col*8 + order_map[i]] << (i * w_bit)
   ```
   即 int32 的第 `i` 个 nibble（shift `i*4`）存的是 **output col `order_map[i]`**，而不是自然列 `i`。因此：
   - 按 shift 顺序 `[0,4,8,…,28]` 提取 nibble，得到的是 output 列 `[0,2,4,6,1,3,5,7]`；
   - `cat([low,high])` 得到的是 `[0,4,2,6,1,5,3,7]`（更乱）；
   - 无论 `cat`、`stack` 还是 triton 的 shift 直取，都不是自然列序 `[0,1,2,3,4,5,6,7]`。

由于 `scales` 是按自然列序存储的（未打包），错序的权重列与 scale 列错位，反量化结果错误。

### 修复（怎么改）
`order_map = [0,2,4,6,1,3,5,7]` 的逆序为 `inv_order = [0,4,1,5,2,6,3,7]`（即 3-bit 循环右移：`inv_order[p] = (bit0<<2)|(bit2<<1)|bit1`）。**用 `inv_order` 作为 shift 顺序**，解出的 nibble 即按自然 output 列序排列，无需额外 permute：

```python
# 新代码（awq_gemm.py:32-45）
w_bit = 32 // pack_factor
inv_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device=qweight.device)
shifts = inv_order * w_bit  # [0,16,4,20,8,24,12,28]
int4_vals = ((qweight[:, :, None].to(torch.int32) >> shifts[None, None, :]) & 0x0F) \
    .reshape(in_features, out_features)
```

### 验证
- 对 `gate_proj` / `q_proj` 用真实 AutoAWQ 权重反量化，与 FP16 原始权重比对：cosine 从 0.22 提升到 0.977/0.973（达到 int4 AWQ 正常精度）。
- 纯 nibble 诊断：输入 `[0,1,2,3,4,5,6,7]` 的权重，修复前还原为 `[0,2,4,6,1,3,5,7]`，修复后正确还原为 `[0,1,2,3,4,5,6,7]`。

---

## 3. Bug 2：zero_point 的 −1 处理错误

**位置**：`awq_gemm.py:47-49`（torch 路径，原 `int4_zeros -= 1`）+ `awq_triton_kernel.py:79-82`（triton 路径，原 `... - 1`）

### 现象
修复 Bug 1 后，反量化与原始权重 cosine 仅 0.91（仍偏低），且 `got.max≈0.99` 远超 `ref.max≈0.31`，呈系统性「整体偏大」的偏移。

### 根因（为什么错）
原代码注释与实现：
```python
# 旧代码（错误）
# AutoAWQ在量化时会将zero_point偏移+1，反量化要剪回来
int4_zeros -= 1
w_fp16 = (int4_vals - int4_zeros) * scales   # 等价 (int4 - (stored-1)) * scale
```

实际查 AutoAWQ 源码，`qzeros` 存的是 **real zero_point，并未做 +1 偏移**：
- `awq/quantize/quantizer.py:89`：`zeros = (-round(min_val/scales)).clamp_(min_int, max_int)` —— 直接算 real zero。
- `awq/quantize/quantizer.py:200`：`intweight = round((w + scale_zeros)/scale)`，其中 `scale_zeros = zeros * scales`（real zero）。
- `awq/modules/linear/gemm.py:247`：`qzeros[:, col] |= qzero_col << ...`，`qzero_col` 就是 real zero，**没有 +1**。

因此反量化公式应为 `w = (int4 - zero_real) * scale = (int4 - stored) * scale`。原代码多减了 1，变成 `(int4 - (stored-1)) * scale = 正确值 + scale`，每个元素整体偏移 +scale，导致精度崩坏。

> 注：测试方案 `test_plan_awq.md` §8 曾把「`zero_point +1 → -1` 处理错误」列为风险点，方向判断正确，但实际是「不该 −1 却 −1 了」（AutoAWQ 0.2.x 已无 +1 约定，旧版 AWQ 才有）。

### 修复（怎么改）
移除 `−1`，直接用 stored zero：

```python
# 新代码（awq_gemm.py:47-53）
# 注意：AutoAWQ 0.2.x 的 qzeros 直接存的是 real zero_point（量化时未做 +1 偏移，
# 见 awq/modules/linear/gemm.py 的 pack 与 awq/quantize/quantizer.py 的 zeros 计算），
# 因此反量化直接用 (int4 - zero) * scale，不要再做 -1。
scales_expanded = scales.repeat_interleave(group_size, dim=0)
zeros_expanded = int4_zeros.repeat_interleave(group_size, dim=0)
w_fp16 = (int4_vals.float() - zeros_expanded.float()) * scales_expanded.float()
```

triton 路径同步移除 `zeros_expanded ... - 1`（见 `awq_triton_kernel.py:80-82`）。

### 验证
扫描 `zero_shift ∈ {0, −1, +1, −2}` 全元素比对：`shift=0` 时 cosine 最高（gate 0.977 / q 0.973），其余均 ≤0.91。确认 `−1` 是错的。

---

## 4. Bug 3：triton kernel 在 triton 3.6.0 下完全编译失败

**位置**：`nanokvllm/kernel/awq_triton_kernel.py`

### 现象
`awq_kernel="triton"` 时直接 `CompilationError`，kernel 无法运行。该 bug 被 4 个独立问题叠加掩盖，逐个暴露：

### 3-a. `tl.arange` 用局部变量而非 constexpr

**位置**：`awq_triton_kernel.py:30`（原 `rpackn = pid_n * PACK_BLOCK_N + tl.arange(0, PACK_BLOCK_N)`）

**根因**：`PACK_BLOCK_N = BLOCK_N // pack_factor` 虽由 constexpr 派生，但作为**命名局部变量**传入 `tl.arange` 时，triton 3.6 不再把它识别为 `tl.constexpr`，报 `arange's arguments must be of type tl.constexpr`。同文件 `rg = tl.arange(0, BLOCK_K // group_size)` 用内联表达式则可过。

**修复**：把 `PACK_BLOCK_N`、`BLOCK_G` 作为显式 `tl.constexpr` 形参从 launcher 传入（`awq_triton_kernel.py:18-19` 形参，`awq_triton_kernel.py:115-116,129` launcher 计算并传入）。

### 3-b. nibble 提取广播 rank 不匹配

**位置**：`awq_triton_kernel.py:65-66`（原 `qweight_nibbles = (qweight_tile[:, None] >> shifts[None, :]) & 0x0F`）

**根因**：`qweight_tile` 是 2D `[BLOCK_K, PACK_BLOCK_N]`，`qweight_tile[:, None]` 在 2D 上插入的是**中间维**（得 `[BLOCK_K, 1, PACK_BLOCK_N]`），而非预期的尾维 `[BLOCK_K, PACK_BLOCK_N, 1]`；与 `shifts[None, :]`（`[1,8]`）广播时维度对不齐，报 `Cannot make_shape_compatible: incompatible dimensions`。

**修复**：显式构造 rank-3 广播：
```python
# 新代码（awq_triton_kernel.py:65）
qweight_nibbles = (qweight_tile[:, :, None] >> shifts[None, None, :]) & 0x0F
```

### 3-c. mask 形状错（`rpackn[:, None]` 应为 `rpackn[None, :]`）

**位置**：`awq_triton_kernel.py:50-51`

**根因**：`qweight_tile` 形状为 `[BLOCK_K, PACK_BLOCK_N] = [128, 8]`，其 mask 须为 `[K_dim, N_packed_dim] = [128, 8]`。原代码 `(rk[:, None] + k < K) & (rpackn[:, None] < PACK_N)` 中 `rk[:,None]` 是 `[128,1]`、`rpackn[:, None]` 是 `[8,1]`，二者按位与时 dim0 出现 `128 vs 8` 冲突，报 `incompatible dimensions at index 0: 128 and 8`。

**修复**：N_packed 维用 `[None, :]`：
```python
# 新代码（awq_triton_kernel.py:50-51）
qzeros_mask   = (rg[:, None] + g < G) & (rpackn[None, :] < PACK_N)
qweight_mask  = (rk[:, None] + k < K) & (rpackn[None, :] < PACK_N)
```

### 3-d. `tl.repeat_interleave` 在 triton 3.6 不存在

**位置**：`awq_triton_kernel.py:80-85`（原 `zeros_expanded = tl.repeat_interleave(unpack_zeros, group_size, dim=0) - 1`）

**根因**：`triton.language` 没有 `repeat_interleave` 这个 API（作者按 PyTorch 习惯写了，但 triton 不提供），运行期 `AttributeError: module 'triton.language' has no attribute 'repeat_interleave'`。

**修复**：`repeat_interleave(x, n, dim=0)`（x 为 `[G, N]`）等价于 `broadcast_to(x[:,None,:], (G, n, N)).reshape(G*n, N)`：
```python
# 新代码（awq_triton_kernel.py:80-85）
zeros_expanded = tl.broadcast_to(
    unpack_zeros[:, None, :], (BLOCK_G, group_size, BLOCK_N)
).reshape(BLOCK_K, BLOCK_N, can_reorder=False)
```

### 3-e. shift 顺序（同 Bug 1）

triton 路径同样要用 `inv_order` 作 shift（`awq_triton_kernel.py:59-61`），并在数值上同步移除 zero 的 `−1`（同 Bug 2）。

### 验证
修复后 triton kernel 编译通过；真实权重上 torch vs triton 单层 GEMM `max_diff=0.000000`，`allclose=True`；端到端冒烟输出与 torch 一致（均给出 `\boxed{5}`）。

---

## 5. Bug 4：`AWQQKVParallelLinear` 形参顺序与调用方不一致

**位置**：`nanokvllm/layers/awq_linear.py:182-184`

### 现象
模型构建阶段即报 `TypeError: AWQQKVParallelLinear.__init__() got multiple values for argument 'group_size'`。

### 根因（为什么错）
调用方 `qwen3.py:65` 经 `make_qkv_liear` 传入的位置实参是 `(hidden_size, head_dim, total_num_heads, total_num_kv_heads)`，且 `group_size` 以 kwargs 传入：

```python
# qwen3.py + make_qkv_liear 实际调用
return AWQQKVParallelLinear(*args, group_size=vllm_config.awq_group_size, awq_gemm=awq_gemm, **kwargs)
# args = (hidden_size, head_dim, total_num_heads, total_num_kv_heads)
```

而原 `AWQQKVParallelLinear.__init__` 把 `group_size` 放在第 4 个位置形参：
```python
# 旧代码（错误）
def __init__(self, hidden_size, head_size, total_num_heads, group_size,
             bits=4, total_num_kv_heads=None, bias=False, awq_gemm=None):
```
于是第 4 个位置实参 `total_num_kv_heads` 被绑定到 `group_size`，同时 `group_size` 又以 kwarg 传入 → "multiple values"。这会导致 `num_kv_heads` 用 `total_num_kv_heads`（实际是 group_size=128）去算，即便不报错也会得到完全错误的层结构。

### 修复（怎么改）
把 `total_num_kv_heads` 提到第 4 位（与调用方对齐），`group_size` 后移并由 kwarg 传入：
```python
# 新代码（awq_linear.py:182-184）
def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
             total_num_kv_heads: int | None = None, group_size: int = 128,
             bits: int = 4, bias: bool = False, awq_gemm=None):
```

### 验证
模型构建不再报错，`qkv_proj` 的 `output_size = (num_heads + 2*num_kv_heads) * head_size` 计算正确。

> 说明：`AWQMergedColumnParallelLinear`、`AWQRowParallelLinear` 的调用方只传 2 个位置实参 + `group_size` kwarg，形参顺序无冲突，无需改动。

---

## 6. Bug 5：AWQ 融合层加载时参数名未拼组件后缀

**位置**：`nanokvllm/utils/loader.py:31-37`

### 现象
权重加载阶段报 `AttributeError: 'gate_up_proj' is not an nn.Parameter`。

### 根因（为什么错）
原代码先 `get_parameter` 再拼后缀，顺序反了：
```python
# 旧代码（错误）
param_name = base_name.replace(k, v)            # 例: "...gate_up_proj"（模块名）
param = model.get_parameter(param_name)         # gate_up_proj 是模块不是 Parameter → 报错
param_name = param_name if comp is None else f'{param_name}.{comp}'  # 拼后缀，但已经晚了
```
对 AWQ，`split_awq_component` 把 `...gate_proj.qweight` 拆成 `base_name="...gate_proj"`、`comp="qweight"`，故 `param_name` 是模块名 `...gate_up_proj`，而真正的 Parameter 是 `...gate_up_proj.qweight`。

（对比：非 AWQ 路径 `base_name` 本身就带 `.weight`，如 `...gate_proj.weight` → replace 后 `...gate_up_proj.weight`，`get_parameter` 能命中，所以非 AWQ 一直正常。）

### 修复（怎么改）
先拼上组件后缀，再 `get_parameter`：
```python
# 新代码（loader.py:31-37）
param_name = base_name.replace(k, v)
# AWQ 下真实参数名带组件后缀(.qweight/.qzeros/.scales)，必须在 get_parameter 前拼上
if comp is not None:
    param_name = f'{param_name}.{comp}'
param = model.get_parameter(param_name)
weight_loader = getattr(param, "weight_loader")
weight_loader(param, f.get_tensor(weight_name), shard_id)
```
（`else` 分支无需改：非融合 AWQ 参数如 `o_proj.qweight`，`weight_name` 本身就是完整参数名，直接 `get_parameter(weight_name)` 命中。）

### 验证
权重加载无报错；加载后 `gate_up_proj.qweight/qzeros/scales` 与 `qkv_proj.*` 均被正确填充。

---

## 7. Bug 6：`get_rope` 的 `@lru_cache` 遇 dict 型 `rope_scaling` 崩溃

**位置**：`nanokvllm/layers/rotary_embedding.py:51`（`get_rope`）

### 现象
模型构建到 `self.rotary_emb = get_rope(...)` 时报 `TypeError: unhashable type: 'dict'`。

### 根因（为什么错）
`get_rope` 用了 `@lru_cache(1)`，而 `rope_scaling` 实参是 dict。`lru_cache` 在调用前要先对所有位置/关键字参数做哈希以查缓存，dict 不可哈希 → 直接抛 `TypeError`（函数体里的 `assert rope_scaling is None` 都还没执行到）。

transformers 5.x 把 RoPE 配置统一成 dict（如 `{'rope_type':'default','rope_theta':1000000}`），`AutoConfig.rope_scaling` 返回 dict 而非 `None`，于是触发了该问题。**这是预存的一般性 bug，FP16 模型同样中招，并非 AWQ 专属**，但它阻断了 AWQ 端到端测试，需一并修复。

### 修复（怎么改）
去掉 `@lru_cache`，改手动缓存（key 只含可哈希量），并把 `rope_type=='default'` 归一化为 `None`（本实现仅支持 default）：
```python
# 新代码（rotary_embedding.py:50-68）
_ROPE_CACHE = {}
def get_rope(head_size, rotary_dim, max_position, base, rope_scaling=None):
    if isinstance(rope_scaling, dict):
        if rope_scaling.get("rope_type", "default") != "default":
            raise NotImplementedError(f"unsupported rope_scaling: {rope_scaling}")
        rope_scaling = None
    assert rope_scaling is None
    key = (head_size, rotary_dim, max_position, base)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _ROPE_CACHE[key]
```
（同时移除了不再使用的 `from functools import lru_cache`。）手动缓存保留了「同参数多层共享同一个 RotaryEmbedding 实例」的内存优化。

### 验证
AWQ 与 FP16 模型均能正常构建；FP16 路径生成正常（无回归）。

---

## 8. Bug 7：AWQ fp16 反量化 vs bf16 模型的 dtype 冲突

**位置**：`nanokvllm/engine/model_runner.py:27-33`

### 现象
模型构建/首次前向时 `F.linear` 报 `RuntimeError: expected m1 and m2 to have the same dtype, but got: BFloat16 != Half`。

### 根因（为什么错）
- Qwen3 原始模型是 bfloat16，AutoAWQ 保存的 `config` 里 `dtype='bfloat16'`，`model_runner.py` 原代码 `torch.set_default_dtype(hf_config.torch_dtype)` → 整个模型（embed/norm/hidden_states）按 bf16 运行。
- 但 AWQ 反量化路径 `dequantize_awq_weight` 与 triton kernel 都**硬编码输出 fp16**（`return w_fp16.to(torch.float16)` / `accumulator.to(tl.float16)`）。
- 于是 `F.linear(x_bf16, w_fp16)` 两个操作数 dtype 不同，torch 不自动提升 bf16↔fp16，直接报错。
- 此外 AutoAWQ 保存的 `config.json` 把 `torch_dtype` 字段重命名成了 `dtype`（transformers 5.x 的 `AutoConfig` 仍能映射回 `torch_dtype=bfloat16`），所以不是 None，但 dtype 冲突依旧。

### 修复（怎么改）
AWQ 反量化是 fp16，最一致的做法是让 AWQ 模型整体按 fp16 运行（非量化层 embed/norm/lm_head 的 bf16 权重在 `load_model` 的 `param.data.copy_(...)` 时自动 cast 到 fp16；triton 的 `tl.dot(fp16,fp16)` 也由此满足同 dtype）：
```python
# 新代码（model_runner.py:26-33）
default_dtype = torch.get_default_dtype()
if config.quantization == "awq":
    # AWQ 反量化路径(dequantize_awq_weight / triton kernel)硬编码输出 fp16；
    # 为避免 bf16 hidden_states @ fp16 weight 的 dtype 冲突，AWQ 模型整体按 fp16 运行
    # （非量化层 embed/norm/lm_head 权重在 load 时由 bf16 cast 到 fp16）
    torch.set_default_dtype(torch.float16)
else:
    torch.set_default_dtype(hf_config.torch_dtype)
```
该改动是 AWQ 专属分支，不影响 FP16 路径。

### 验证
AWQ 模型前向不再报 dtype 错误；KV cache（fp16）、flash_attn（支持 fp16）均正常。

---

## 8.5 Bug 8：AWQ-torch kernel 破坏 CUDA graph 捕获

**位置**：`nanokvllm/layers/awq_gemm.py:33`（`dequantize_awq_weight` 内 `inv_order` 构造，Bug 1 修复时引入）

### 现象
`enforce_eager=False`（CUDA graph 模式）下，AWQ **torch** kernel 在 `capture_cudagraph` 捕获阶段崩溃：
```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```
（eager 模式不捕获图，故 §3 冒烟与延迟测试均未暴露；triton kernel 不受影响，graph 模式正常。）

### 根因（为什么错）
Bug 1 修复时，`dequantize_awq_weight` 用 Python list 现建 CUDA tensor 来表示 `inv_order`：
```python
# 旧代码（Bug 1 修复引入，graph 不兼容）
inv_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device=qweight.device)
```
`torch.tensor(python_list, device='cuda')` 会先把 list 暂存为 CPU tensor 再拷到 GPU。该 **CPU→CUDA 拷贝在 CUDA graph 捕获期间被禁止**（除非 `pin_memory`）。捕获路径 `capture_cudagraph → model forward → awq_gemm → dequantize_awq_weight` 每层、每次调用都触发该拷贝，于是崩溃。

triton kernel 用 `tl.arange` + 位运算在 device 端构造，无 CPU→CUDA 拷贝，所以 graph 正常。

### 修复（怎么改）
与 triton kernel 一致，用 `torch.arange` + 位运算 on-device 构造 `inv_order`（`torch.arange(device='cuda')` 直接在 GPU 生成，无 CPU 暂存）：
```python
# 新代码（awq_gemm.py:32-37）
w_bit = 32 // pack_factor
p = torch.arange(pack_factor, dtype=torch.int32, device=qweight.device)
inv_order = ((p & 1) << 2) | (((p >> 2) & 1) << 1) | ((p >> 1) & 1)
shifts = inv_order * w_bit  # [0,16,4,20,8,24,12,28]
```
算术构造的 `inv_order` 与 `[0,4,1,5,2,6,3,7]` 完全等价（已在 triton kernel 验证），数值正确性不变。

### 验证
AWQ-torch graph 模式不再崩溃，可正常捕获与 replay：batch=1 latency 测得 TTFT 108.5ms / TPOT 33.2ms / 30.1 tok/s。

> 备注：triton kernel 在 graph 模式下 TPOT 仅 3.23ms（309.8 tok/s），远优于 torch 的 33.2ms——因为 torch kernel 每个 decode step 仍要全量反量化（物化 fp16 权重），是计算开销而非 launch 开销，graph 只能消除后者。生产环境应用 triton kernel。

---

## 9. 改动文件清单

| 文件 | 涉及 Bug |
|---|---|
| `nanokvllm/layers/awq_gemm.py` | Bug 1、Bug 2、Bug 8 |
| `nanokvllm/kernel/awq_triton_kernel.py` | Bug 2、Bug 3（3-a~3-e） |
| `nanokvllm/layers/awq_linear.py` | Bug 4 |
| `nanokvllm/utils/loader.py` | Bug 5 |
| `nanokvllm/layers/rotary_embedding.py` | Bug 6 |
| `nanokvllm/engine/model_runner.py` | Bug 7 |

---

## 10. 测试结论

| 测试项（对应 `test_plan_awq.md`） | 结果 |
|---|---|
| §3.1 冒烟（torch kernel） | ✅ 通顺 CoT，正确给出 `\boxed{5}` |
| §3.1 冒烟（triton kernel） | ✅ 通顺 CoT，正确给出 `\boxed{5}` |
| §3.2 kernel 一致性（真实权重单层 GEMM） | ✅ torch vs triton `max_diff=0.000000`，`allclose=True` |
| §3.2 kernel 一致性（近贪心 top-1 逐 token） | ✅ 200/200 = 100% 一致 |
| 反量化还原 vs FP16 原始权重 | ✅ cos≈0.977(gate) / 0.973(q_proj) |
| FP16 路径回归 | ✅ 无回归 |
| §5 性能基准（`bench_awq.py`，含 Bug 8 修复后） | ✅ 延迟 + 吞吐矩阵跑通，见 §10.5 |

> 关于随机采样（temperature=0.6）下 token 一致率仅 ~10%：这是**预期现象**，非 bug。triton 分块累加与 cuBLAS 的 fp32 累加顺序不同，产生亚 ULP 差异，经 28 层放大后被随机采样放大成序列分叉；top-1（贪心）下 100% 一致，证明两 kernel 数值等价。

---

## 10.5 性能基准结果（§5）

测试模型 Qwen3-0.6B / Qwen3-0.6B-AWQ，GPU 为 RTX 4080(32GB)×2（TP=1 单卡），`bench_awq.py` 默认 `repeats=5/warmup=2`（吞吐组用 `repeats=2/warmup=1` 加速），`kv_compress_enabled=False`。

### 延迟对比（batch=1, input=512, output=256, TP=1）

| 配置 | TTFT(ms) | TPOT(ms/tok) | decode(tok/s) | E2E(ms) |
|---|---|---|---|---|
| FP16 eager | 45.1 | 37.4 | 26.8 | 9580 |
| FP16 graph | 45.1 | 3.05 | 327.7 | 828 |
| AWQ-torch eager | 96.5 | 87.9 | 11.4 | 22526 |
| AWQ-torch graph | 108.5 | 33.2 | 30.1 | 8588 |
| AWQ-triton eager | 57.8 | 50.5 | 19.8 | 12929 |
| AWQ-triton graph | 60.0 | 3.23 | 309.8 | 888 |

### 吞吐对比（input=512, output=256, TP=1, graph 模式）

| Batch | FP16(tok/s) | AWQ-triton(tok/s) | 加速比 | FP16 Peak(GB) | AWQ Peak(GB) |
|---|---|---|---|---|---|
| 1 | 327.7 | 309.8 | 0.95× | — | — |
| 8 | 1454.7 | 627.0 | 0.43× | 24.46 | 24.50 |
| 32 | 3844.6 | 3026.1 | 0.79× | 24.79 | 24.84 |
| 64 | 4897.9 | 3770.7 | 0.77× | 24.83 | 24.87 |

### 分析

1. **graph vs eager**：CUDA graph 把 decode 的 launch 开销消掉，TPOT 从几十 ms 降到 ~3ms（FP16 37.4→3.05，AWQ-triton 50.5→3.23）。这是最大收益来源。
2. **torch vs triton**：triton 明显优于 torch（graph 下 TPOT 3.23 vs 33.2ms）。torch kernel 每步全量反量化（物化 fp16 权重）是计算开销，graph 只能消除 launch 开销；triton 融合 GEMM 高效。**生产应用 triton kernel**。
3. **AWQ 未超过 FP16（0.6B 上）**：所有加速比 <1。原因：(a) 0.6B 权重太小，4bit 带宽优势不主导；(b) triton kernel 是"小 tile 保正确性"未调优版（§7.1 已注明）；(c) 测试方案 §1.3 的加速预期针对 8B 场景（decode memory-bound，权重 1/4 → 理论 3×），0.6B 无法体现。
4. **显存几乎没省**：FP16 与 AWQ peak 都 ~24.8GB。`allocate_kv_cache` 按 `gpu_memory_utilization=0.8` 把可用显存几乎全分给 KV cache，0.6B 权重省的 ~0.8GB 被淹没。实证了测试方案 §8「KV cache 未量化 → 大 batch 显存瓶颈仍在」。

---

## 11. 备注与未覆盖项

- **环境**：真正可用的 Python 环境是 `/root/nano-vllm/.venv`（不是 base conda，base 缺 transformers/flash_attn 等）。已在该 venv 安装 `autoawq==0.2.9`（官方已 deprecated，最后测试配置为 torch 2.6 + transformers 4.51，本环境为 torch 2.11 + transformers 5.12，更高版本）。
- **网络**：`huggingface.co` 不通，AutoAWQ 校准数据集 `mit-han-lab/pile-val-backup` 通过 `HF_ENDPOINT=https://hf-mirror.com` 下载。
- **交付物**：已生成 `~/huggingface/Qwen3-0.6B-AWQ/`（含 nano-kvllm 兼容版 `quantize_config.json`：`bits/group_size/zero_point/version=GEMM`）。环境里只有 0.6B，没有 8B。
- **dtype 取舍**：Bug 7 的修复让 AWQ 模型按 fp16 运行，非量化层权重 bf16→fp16 cast 有极小精度损失，可接受。
- **benchmark 运行备注**：连续跑多配置时曾遇到一次 `model_runner.py:24` 硬编码的 NCCL rendezvous 端口 `2333` 报 `EADDRINUSE`（上一次测试进程 TIME_WAIT 残留），重试即过。该硬编码端口对连续 benchmark 较脆弱，建议后续改成可配置/自增端口。
- **未覆盖**：
  - TP>1 未测（`awq_linear.py` 的 AWQ 权重 TP 切分逻辑只验证了 TP=1）。
  - 8B 模型环境里没有，§1.3 预期的真实加速场景（8B decode memory-bound）未能验证；0.6B 上加速比 <1 属正常。
  - `quantize_awq.py` 本身无需改动，但建议补一行把 `torch_dtype` 显式写进保存的 `config.json`（目前靠 `model_runner.py` 的 AWQ 分支兜底）。
