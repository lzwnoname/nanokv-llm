# AWQ 集成代码 Bug 清单与修复指南

> 审查范围：`config.py` / `utils/loader.py` / `layers/awq_linear.py` / `layers/awq_gemm.py` / `models/qwen3.py`
> 审查时间：2026-07-08
> 当前进度：Phase 1~2（Triton kernel 未实现，不影响 Bug 判定）

---

## 一、严重 Bug（直接导致运行报错）

### Bug 1：`config.py` 属性名拼写错误

```python
# 第 21 行
quatization: str | None = None    # ← 少了 "n"
```

`__post_init__` 第 42 行用的是 `self.quantization`，会报 `AttributeError`。

**修复**：`quatization` → `quantization`。

---

### Bug 2：`qwen3.py` 引用不存在的 config 属性

```python
# 第 17 行
awq_gemm = get_awq_gemm(vllm_config.kernel)              # ← 应为 vllm_config.awq_kernel
return AWQMergedColumnParallelLinear(..., group_size=vllm_config.group_size, ...)  # ← 应为 awq_group_size
# 第 24 行同理
```

`vllm_config.kernel` 和 `vllm_config.group_size` 在 `Config` 里都不存在。

**修复**：
- `vllm_config.kernel` → `vllm_config.awq_kernel`
- `vllm_config.group_size` → `vllm_config.awq_group_size`

---

### Bug 3：`qwen3.py` 引用未赋值的 `self.vllm_config`

```python
# Qwen3Attention.__init__ 第 58-59 行
self.qkv_proj = make_qkv_liear(self.vllm_config, ...)   # ← self.vllm_config 从未赋值
# Qwen3MLP.__init__ 第 120-121 行同理
self.gate_up_proj = make_column_liear(self.vllm_config, ...)
```

`Qwen3Attention.__init__` 和 `Qwen3MLP.__init__` 都没有 `self.vllm_config = vllm_config` 赋值语句。

更严重的是 `Qwen3MLP.__init__` 的参数列表里**根本没有接收 `vllm_config`**：

```python
# 第 113-118 行
def __init__(self, hidden_size, intermediate_size, hidden_act):   # ← 缺 vllm_config
```

而 `Qwen3DecoderLayer` 调用时也没传：

```python
# 第 162-166 行
self.mlp = Qwen3MLP(
    hidden_size=config.hidden_size,
    intermediate_size=config.intermediate_size,
    hidden_act=config.hidden_act,
    # ← 缺 vllm_config
)
```

**修复**：
1. `Qwen3Attention.__init__` 和 `Qwen3MLP.__init__` 都加 `self.vllm_config = vllm_config`
2. `Qwen3MLP.__init__` 参数列表加 `vllm_config`
3. `Qwen3DecoderLayer` 调用 `Qwen3MLP` 时传入 `vllm_config`

---

### Bug 4：`awq_linear.py` `dist.getrank()` 拼写错误

```python
# 第 19 行
self.tp_rank = dist.getrank()    # ← 应为 dist.get_rank()
```

**修复**：`dist.getrank()` → `dist.get_rank()`。

---

### Bug 5：`awq_linear.py` `copy` 少下划线（3 处）

```python
# 第 47 行（weight_loader_qweight）
param.data.copy(loaded_weight)    # ← 应为 copy_
# 第 50 行（weight_loader_qzeros）
param.data.copy(loaded_weight)
# 第 53 行（weight_loader_scales）
param.data.copy(loaded_weight)
```

`Tensor.copy` 不是 in-place 方法，`copy_` 才是。

**修复**：三处 `.copy(` → `.copy_(`。

---

### Bug 6：`awq_linear.py` `AWQColumnParallelLinear` 未传 `tp_dim`

```python
# 第 64 行
super().__init__(input_size, divide(output_size, tp_size), group_size, bits, bias, awq_gemm)
# AWQLinear.__init__ 签名：(input_size, output_size, group_size, bits, bias, tp_dim, awq_gemm)
# 当前调用：bias 后直接传 awq_gemm，跳过了 tp_dim
```

**修复**：补传 `tp_dim=1`（AWQ qweight 是 `[in, out//pack]`，output 在 dim=1）：

```python
super().__init__(input_size, divide(output_size, tp_size), group_size, bits, bias, tp_dim=1, awq_gemm=awq_gemm)
```

---

### Bug 7：`awq_linear.py` `self.output_sizes` 赋值错误

```python
# 第 92 行（AWQMergedColumnParallelLinear.__init__）
self.output_sizes = self.output_sizes    # ← 右侧引用了未定义的属性
```

**修复**：`self.output_sizes = output_sizes`（用参数名）。

---

### Bug 8：`awq_gemm.py` `dequantize_awq_weight` 未定义

```python
# 第 8 行
w_fp16 = dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor)
```

整个文件里没有定义这个函数。

**修复**：在 `awq_gemm.py` 里实现 `dequantize_awq_weight` 函数（详见本文档第二部分"反量化实现方案"）。

---

### Bug 9：`awq_gemm.py` `return NotImplementedError` 应为 `raise`

```python
# 第 25 行
return NotImplementedError    # ← 应为 raise
```

`return NotImplementedError` 会把 `NotImplementedError` 类本身当作返回值返回，不会抛异常。

**修复**：`return` → `raise`。

---

## 二、逻辑 Bug（不报错但结果错误）

### Bug 10：`awq_linear.py` chunk 索引用了 `loaded_shard_id` 而非 `self.tp_rank`

```python
# AWQMergedColumnParallelLinear 第 103 行
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[loaded_shard_id]
# ← loaded_shard_id 是子模块编号(0/1)，不是 rank 编号
# qzeros（第 114 行）和 scales（第 125 行）同理
```

对比原始 `MergedColumnParallelLinear.weight_loader`：

```python
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]   # ← 用 tp_rank
```

**修复**：三处 `[loaded_shard_id]` → `[self.tp_rank]`。

---

### Bug 11：`loader.py` AWQ 分支用 `weight_name` 而非 `base_name` 做匹配

```python
# 第 29-33 行
if k in weight_name:                                    # ← 应为 base_name
    v, shard_id = packed_modules_mapping[k]
    param_name = weight_name.replace(k, v)              # ← 应为 base_name
    param = model.get_parameter(param_name)             # ← 用的是没加 comp 后缀的名字
    param_name = param_name if comp is None else f'{param_name}.{comp}'  # ← 重新赋值后未被使用
```

当前逻辑虽然不会直接报错（`get_parameter` 在重新赋值前执行了），但 `weight_name` 带 `.qweight` 后缀时匹配/替换语义混乱。

**修复**：统一用 `base_name`：

```python
if k in base_name:
    v, shard_id = packed_modules_mapping[k]
    param_name = base_name.replace(k, v)
    param_name = param_name if comp is None else f'{param_name}.{comp}'
    param = model.get_parameter(param_name)
    weight_loader = getattr(param, "weight_loader")
    weight_loader(param, f.get_tensor(weight_name), shard_id)
    break
```

---

### Bug 12：`qwen3.py` `o_proj` 和 `down_proj` 未替换为 AWQ 版本

```python
# 第 66 行
self.o_proj = RowParallelLinear(...)        # ← 仍是 FP16 版
# 第 126 行
self.down_proj = RowParallelLinear(...)     # ← 仍是 FP16 版
```

如果 checkpoint 里量了 `o_proj`/`down_proj`（通常会），加载时会因模型里没有 `qweight` 参数而报错。

**修复**：新增 `make_row_linear` 工厂函数 + `AWQRowParallelLinear` 类，替换两处调用。

---

### Bug 13：`config.py` checkpoint 文件名与 AutoAWQ 标准不一致

```python
# 第 43 行
quant_config_path = os.path.join(self.model, "quant_config.json")
```

AutoAWQ 标准产物文件名是 `quantize_config.json`（多了个 `e`）。

**修复**：确认实际使用的文件名，统一为 `quantize_config.json` 或显式支持两者。

---

## 三、拼写/命名汇总

| 文件 | 行 | 错误 | 修正 |
|---|---|---|---|
| `config.py` | 21 | `quatization` | `quantization` |
| `config.py` | 25 | 注释 `trition` | `triton` |
| `awq_linear.py` | 19 | `dist.getrank()` | `dist.get_rank()` |
| `awq_linear.py` | 47,50,53 | `.copy(` | `.copy_(` |
| `awq_gemm.py` | 12 | `AWQGemmTrition` | `AWQGemmTriton` |
| `awq_gemm.py` | 15 | `awq_trition_kernel` | `awq_triton_kernel` |
| `awq_gemm.py` | 22 | `"trition"` | `"triton"` |
| `awq_gemm.py` | 25 | `return NotImplementedError` | `raise NotImplementedError` |

---

## 四、结构遗漏

### 遗漏 1：缺少 `AWQRowParallelLinear`

`o_proj` 和 `down_proj` 是 RowParallel（沿 input 维切分），当前只有 ColumnParallel 系列的 AWQ 实现，没有 RowParallel 版本。

需要新增 `AWQRowParallelLinear`，要点：
- input 维（qweight 的 dim=0）按 TP 切分
- 切分需对齐 `group_size`（因为 qzeros/scales 沿 input 维有 group 结构）
- `forward` 里保留 `all_reduce` 逻辑

### 遗漏 2：缺少 `make_row_linear` 工厂函数

`qwen3.py` 里 `o_proj` 和 `down_proj` 需要通过工厂函数按 `quantization` 开关分发到 `AWQRowParallelLinear` 或 `RowParallelLinear`。

### 遗漏 3：`AWQColumnParallelLinear` 的 weight_loader 是死代码

普通 `ColumnParallelLinear` 不在 `packed_modules_mapping` 里，不会被 loader 走 packed 分支命中。其 weight_loader 定义了 `loaded_shard_id` 参数但永远不会被调用。作为父类被继承时被子类覆盖，不影响功能，但可以清理或加注释说明。

---

## 五、修复优先级

| 优先级 | Bug 编号 | 说明 |
|---|---|---|
| P0（不修无法运行） | 1,2,3,4,5,6,7,9 | 拼写/签名/赋值错误 |
| P1（不修结果错误） | 8,10,11,12,13 | 逻辑/加载/遗漏 |
| P2（清理项） | 遗漏 1,2,3 | 结构补全 |

建议按 P0 → P1 → P2 顺序修复，每修完一批跑一次"加载 + 单条 prompt 推理"做冒烟测试。
