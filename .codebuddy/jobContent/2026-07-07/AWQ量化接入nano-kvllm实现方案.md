# AWQ 量化接入 nano-kvllm 实现方案

> 目标：在 `nano-kvllm` 中引入 **AWQ（Activation-aware Weight Quantization）INT4 权重量化**，降低模型权重显存占用（约 3~4x），并在 decode（memory-bound）阶段获得推理加速；同时保持与现有 **PagedAttention / TP / 窗口化周期性 KV 压缩** 机制正交、不冲突。

---

## 1. AWQ 技术原理

### 1.1 核心问题：权重量化的精度损失从哪来

LLM 权重量化（weight-only quantization）把 FP16 权重压成 INT4/INT8，主要目的不是省计算量，而是**省显存带宽**——decode 阶段是逐 token 生成，矩阵乘法是 GEMV（矮胖矩阵×向量），计算强度低，瓶颈是"从显存搬权重"，权重体积越小，搬得越快，decode 越快。

直接量化的问题：weight 分布里少数"离群值/重要通道"对输出误差贡献极大，粗暴 round-to-nearest 量化会让这批通道产生较大误差，进而在下游放大。

### 1.2 AWQ 的核心洞察：按激活值定位"重要通道"，而非按权重

AWQ（[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)，MIT 韩松组）的关键发现：

- **决定一个权重通道是否"重要"的，不是这个通道权重本身的幅值大小，而是这个通道对应的输入激活（activation）的幅值大小**。
- 用少量校准数据（几百条样本）跑一次前向，统计每个 input channel 的激活幅值分布，幅值大的 channel → 该 channel 对应的权重列是"salient/显著"权重。
- 实验表明：只要保护住这 ~1% 的显著权重（不量化或降低其量化误差），量化后整体误差就能大幅降低——而不需要像 GPTQ 那样对所有权重做基于 Hessian 的逐权重补偿。

### 1.3 如何"保护"显著权重：等价的 per-channel scaling，而不是混合精度

最直接的想法是把显著权重保持 FP16、其余权重量化为 INT4——但这是**混合精度**，在硬件上难以实现高效 kernel（需要额外的稀疏/选择逻辑）。

AWQ 给出一个**数学等价变换**，避免真正的混合精度：

对于 `Y = X · W`，引入一个 per-input-channel 的缩放向量 `s`：

```
Y = X · W
  = (X · diag(s)^-1) · (diag(s) · W)
```

- 把**显著 channel 的权重乘以 s（s > 1）放大**，量化时相对误差就变小（量化误差是绝对误差，权重被放大后，`量化误差 / 权重值` 的相对比例下降）；
- 同时把对应的输入激活 `X` 除以 `s`，保证数学上完全等价，不引入额外计算（`s` 可以离线融合进上一层的 LayerNorm/权重里，或者在推理时用一个廉价的逐 channel 乘法完成）；
- 代价：非显著 channel 的量化误差会略微上升，但因为它们本来就不重要，对整体误差影响很小。

**如何求最优的 `s`：** AWQ 不用梯度下降/反向传播（这是它比 GPTQ 更"轻"、更不容易在校准集上过拟合的原因），而是：

```
s = (激活幅值统计量)^alpha
```

对若干候选 `alpha ∈ [0, 1]` 做**网格搜索**，每个候选算出量化后的输出 MSE，选 MSE 最小的 `alpha`。整个校准过程只需要若干次前向（forward-only），非常快（几分钟到几十分钟级别，视模型大小）。

### 1.4 分组量化（group-wise quantization）

AWQ 采用**per-group 非对称量化**（不是简单的 per-tensor）：

- 沿 input channel（即矩阵乘法的 `K` 维）把权重切成若干组，常用 `group_size = 128`；
- 每组独立算 `scale` 和 `zero_point`：
  ```
  scale      = (max(w_group) - min(w_group)) / (2^bits - 1)
  zero_point = round(-min(w_group) / scale)              # clip 到 [0, 2^bits-1]
  qw         = clip(round(w / scale) + zero_point, 0, 2^bits-1)   # INT4 存储值
  ```
- 推理时反量化：`w_fp16 = (qw - zero_point) * scale`（每组共享一对 scale/zero_point）。
- group 越小，精度越高但 scale/zero_point 元数据越多（显存/带宽开销上升），`group_size=128` 是精度/开销的常见平衡点。

### 1.5 与 GPTQ 的对比（帮助定位技术选型）

| | AWQ | GPTQ |
|---|---|---|
| 误差补偿方式 | 激活感知的 per-channel scaling（不改变权重"形状"，只是缩放） | 基于二阶信息（Hessian）逐权重迭代补偿（OBQ/OBC） |
| 是否需要反向传播 | 不需要，纯前向统计 + 网格搜索 | 不需要反向传播，但需要逐层做矩阵求逆等计算，量化过程更重 |
| 校准数据依赖/过拟合风险 | 较低（只统计激活量级分布） | 较高（逐权重拟合校准集的二阶统计量） |
| 量化耗时 | 更快 | 更慢 |
| 精度（社区评测） | 相近，AWQ 在部分模型/低比特下略优 | 相近 |
| 推理形式 | Weight-only INT4，权重打包+分组 scale/zero | 同样是 Weight-only INT4，权重打包+分组 scale/zero（格式不同） |

两者最终落到推理侧都是"INT4 打包权重 + 分组 scale/zero + 运行时反量化做 GEMM"，因此**推理 kernel 层面可以复用同一套设计思路**，只是量化算法（离线生成 qweight 的过程）不同。本方案聚焦 AWQ，但 kernel 设计留有扩展到 GPTQ 的空间。

---

## 2. 业界具体实现调研

### 2.1 AutoAWQ（社区最主流的 AWQ 量化 + 推理库）

- 仓库：`casper-hansen/AutoAWQ`，核心量化线性层实现在 `awq/modules/linear/gemm.py`（`WQLinear_GEMM`），还有 `awq/modules/triton/gemm.py` 的 Triton 版本。
- **权重打包格式**（`WQLinear_GEMM`，4bit，`pack_factor = 32 / 4 = 8`）：
  - `qweight`: `int32`，形状 `[in_features, out_features // pack_factor]`——沿**输出通道**方向，把 8 个 INT4 值打包进 1 个 `int32`（打包顺序是特定的交织顺序 `[0,2,4,6,1,3,5,7]`，服务于 CUDA kernel 的高效位运算解包，不是简单顺序拼接）。
  - `qzeros`: `int32`，形状 `[in_features // group_size, out_features // pack_factor]`，同样打包方式，每个 group 一套 zero_point。
  - `scales`: `fp16`，形状 `[in_features // group_size, out_features]`，每个 group、每个输出通道一个 scale。
  - 配套 `quant_config.json` 记录 `bits`（通常 4）、`group_size`（通常 128）、`zero_point`（是否非对称，AWQ 默认 True）、`version`（`GEMM`/`GEMV`/`Marlin`/`ExLlama` 等 kernel 变体标记）。
- **GEMM vs GEMV 两种 kernel 变体**：
  - `GEMM`：适合 batch size 较大（prefill / 高并发 decode 攒批），用标准分块矩阵乘 + 运行时反量化。
  - `GEMV`：专门为 `batch=1`（或很小 batch）优化的反量化+向量乘，单请求低延迟场景更快。
  - AutoAWQ 官方建议：低并发/单用户走 GEMV，高并发批量走 GEMM。**这与本项目"高并发批量场景"的定位一致，因此本方案优先对接 GEMM 路径。**
- vLLM 对 AWQ 的支持路径类似：`AWQConfig`/`AWQLinearMethod` 在 `create_weights` 阶段创建 `qweight/qzeros/scales` 三个 `Parameter`，并在 `apply()` 里调用 CUDA kernel（`ops.awq_gemm` / 新版 `AWQMarlin` 用 Marlin kernel）做反量化+矩阵乘。**vLLM 文档也明确指出：AWQ 在小并发/低延迟场景收益最明显，大 batch 下吞吐提升有限甚至不如 FP16**（因为大 batch 下计算已经从 memory-bound 转为 compute-bound，量化省下的带宽红利被摊薄）——这一点会写入本方案的"预期收益边界"里，避免过度承诺。

### 2.2 Triton 版实现（更贴合本项目现有技术栈）

- `nano-kvllm` 目前**没有引入任何自定义 CUDA 扩展**，所有自定义算子都用 **Triton**（如 `layers/attention.py::store_kvcache_kernel`）+ `flash-attn` 官方包完成，风格是"轻量、可读、少编译依赖"。
- 社区已有成熟的 **Triton INT4 反量化 + GEMM 融合 kernel** 参考（AutoAWQ 自带的 `awq/modules/triton/gemm.py`，以及论文 *"Accelerating a Triton Fused Kernel for W4A16 Quantized Inference"*（arXiv:2402.00025）思路）：核心做法是**把"反量化"和"矩阵乘"融合进同一个 kernel**（fused dequant+GEMM），避免先反量化出一份完整 FP16 权重再做 `matmul` 带来的额外显存写读开销。
- 这与本项目现有 Triton 使用习惯（自己写 kernel，不依赖预编译 CUDA 扩展）高度契合，是本方案**主推的 kernel 路线**（细节见第 4.4 节）。

### 2.3 现成 CUDA 扩展包（可选的性能兜底方案）

- `autoawq-kernels`（AutoAWQ 配套的预编译 CUDA 扩展）、vLLM 的 `awq_gemm`/`AWQMarlin`（Marlin kernel，社区公认目前 W4A16 最快的通用 kernel之一）。
- 优点：性能成熟、经过大量生产验证；缺点：引入 CUDA 扩展编译/预编译 wheel 依赖，与本项目当前"零 CUDA 扩展、纯 Triton + flash-attn"的轻量风格不符，且会显著增加安装门槛（对齐 CUDA/Torch ABI 版本）。
- **本方案定位为"可选可插拔的性能后端"**，不作为首期必须项（见第 5 节路线图 Phase 3）。

---

## 3. 现状代码分析（改造点定位）

| 文件 | 现状 | 与 AWQ 集成的关系 |
|---|---|---|
| `nanokvllm/config.py` | 无量化相关字段 | 需新增 `quantization` / `awq_group_size` 等配置项 |
| `nanokvllm/utils/loader.py::load_model` | 遍历 safetensors，按 `packed_modules_mapping` 把权重分发给各 `nn.Parameter` 的 `weight_loader`，每个权重名对应**一个** tensor | AWQ checkpoint 每个 Linear 对应**三个** tensor（`qweight/qzeros/scales`），且 QKV/gate_up 融合层需要对三个 tensor 分别做 shard 拼接——**这是核心改造点** |
| `nanokvllm/layers/linear.py` | `LinearBase` 只声明单个 `self.weight`；`ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear`/`MergedColumnParallelLinear` 均基于"整块权重按 TP 维度 narrow/chunk"实现 TP 切分 | 需要新增一套 **AWQ 版本**的这些类（或改造为"按 quant_method 分发"的统一实现），持有 `qweight/qzeros/scales` 三个 Parameter，`forward` 调用量化 GEMM 而不是 `F.linear` |
| `nanokvllm/models/qwen3.py` | `Qwen3Attention`/`Qwen3MLP` 直接实例化 `QKVParallelLinear` 等具体类 | 需要改成"按 config 里的量化开关选择具体 Linear 实现类"（工厂模式），模型结构代码本身尽量不动 |
| `nanokvllm/engine/model_runner.py::allocate_kv_cache` | 按 FP16/BF16 权重估算/分配 KV cache 显存 | 量化只影响**权重**显存，不影响 KV cache dtype，理论上不需要改动，但显存预算会因权重变小而更宽松（可以给 KV cache 分配更多显存 / 支持更大 `num_kvcache_blocks`），是量化带来的直接收益点之一 |
| KV 压缩机制（`compress_utils.py` / `CompressMethod.py`） | 操作对象是 K/V cache（激活值），与权重量化完全独立 | **正交，无需改动**；量化只改权重的存储/计算方式，不改变 Attention 输出的语义，压缩逻辑可以原样复用 |

结论：**核心改造集中在 `linear.py` + `loader.py` + `config.py` 三个文件，`models/qwen3.py` 只需极少量"选择实现类"的改动，KV 压缩相关代码完全不用动。**

---

## 4. 集成方案设计

### 4.1 整体设计原则

1. **量化开关可插拔**：`config.quantization` 为 `None`/`"awq"`，默认关闭，不影响现有 FP16/BF16 推理路径。
2. **模型结构代码不侵入**：`Qwen3Attention`/`Qwen3MLP` 里创建 Linear 层的调用方式基本不变，通过一个小的 **Linear 工厂函数**按 `quantization` 分发到 FP16 版或 AWQ 版实现类，两套类对外接口（构造参数、`forward` 签名）保持一致。
3. **TP 切分规则显式校验**：所有 AWQ Linear 的 shard 维度必须与 `pack_factor`（4bit→8）、`group_size`（默认 128）对齐，否则在 `__post_init__`/构造期直接报错，而不是运行时数值错误。
4. **先正确性、后性能**：分阶段实现（见第 5 节），第一阶段用"反量化整块权重 + `F.linear`"确保正确性和精度可验证，第二阶段再上 Triton fused kernel 提升性能。

### 4.2 Config 扩展

```python
@dataclass
class Config:
    ...
    # ---- AWQ 量化相关 ----
    quantization: str | None = None      # None（不量化）| "awq"
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True          # AWQ 默认非对称量化
    awq_kernel: str = "torch"            # "torch"（反量化+F.linear，正确性优先）| "triton"（fused kernel，性能优先）

    def __post_init__(self):
        ...
        if self.quantization == "awq":
            assert self.awq_bits in (4,), "当前仅支持 4bit AWQ"
            assert self.kvcache_block_size % 256 == 0  # 原有校验不变，量化与 KV block 无关
```

- 读取 checkpoint 目录下的 `quantize_config.json`（AutoAWQ 标准产物）自动填充 `awq_bits/awq_group_size/awq_zero_point`，避免用户手填出错：若目录存在该文件且用户未显式传参，则以文件内容为准；否则报错提示"检测到量化 checkpoint 但缺少 quantize_config.json"。

### 4.3 权重加载改造（`loader.py`）

**判定量化模型**：检查 safetensors 里是否存在 `.qweight` 后缀的 key（或直接依赖 `config.quantization == "awq"`），二者一致性做一次断言校验。

**新的加载分支**：

```python
AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales")

def load_model(model: nn.Module, path: str, quantization: str | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if quantization == "awq":
                    base_name, comp = split_awq_component(weight_name)  # comp ∈ {qweight, qzeros, scales}
                else:
                    base_name, comp = weight_name, None
                for k in packed_modules_mapping:
                    if k in base_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = base_name.replace(k, v)
                        param_name = param_name if comp is None else f"{param_name}.{comp}"
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    ...同理处理非 packed 权重...
```

**关键难点：融合层（QKV / gate_up）的量化权重拼接**

AutoAWQ 量化通常是**逐个原始 Linear 模块**（`q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`）分别量化的，每个模块各自有一套 `qweight/qzeros/scales`。而本项目为了效率把它们**融合**成 `QKVParallelLinear`/`MergedColumnParallelLinear` 一个大矩阵。因此每个 `weight_loader` 在做 shard 拼接时，需要同时对齐三个 tensor 各自的维度语义：

| Tensor | 原始 shard 维度（沿 `output_size`） | 打包后的实际维度 |
|---|---|---|
| `qweight` | `output_size` 维 | `output_size // pack_factor`（8） |
| `qzeros` | `output_size` 维 | `output_size // pack_factor`（8） |
| `scales` | `output_size` 维 | `output_size`（未打包，fp16 存储） |

即 `qweight`/`qzeros` 的 shard offset/size 要先按原始 `output_size` 算好，再整体除以 `pack_factor`；`scales` 直接按原始 `output_size` 切。**必须保证每个子模块（如 `q_proj`）的 `output_size` 是 `pack_factor × TP_size` 的整数倍**，否则无法做无损的按 rank 均匀切分——这是 AWQ 量化 checkpoint 与 TP 并行结合时的一个已知约束，需要在模型加载时显式断言校验，而不是留给运行时报未知错误。

### 4.4 新增量化 Linear 层（`layers/linear.py` 或新文件 `layers/awq_linear.py`）

新增一组与现有类**接口对齐**的量化版本：

```python
class AWQLinearBase(nn.Module):
    def __init__(self, input_size, output_size, group_size, bits=4, bias=False, tp_dim=None):
        super().__init__()
        self.pack_factor = 32 // bits
        self.group_size = group_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.qweight = nn.Parameter(
            torch.empty(input_size, output_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.empty(input_size // group_size, output_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(input_size // group_size, output_size, dtype=torch.float16),
            requires_grad=False,
        )
        self.qweight.weight_loader = self.weight_loader_qweight
        self.qzeros.weight_loader = self.weight_loader_qzeros
        self.scales.weight_loader = self.weight_loader_scales
        ...
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x):
        return awq_gemm(x, self.qweight, self.qzeros, self.scales,
                         self.group_size, self.pack_factor, bias=self.bias)
```

按现有风格分别派生 `AWQColumnParallelLinear`（output 维 shard，对齐 `pack_factor`）、`AWQRowParallelLinear`（input 维 shard，对齐 `group_size`，`forward` 里保留原有 `all_reduce` 逻辑）、`AWQMergedColumnParallelLinear`、`AWQQKVParallelLinear`（`weight_loader` 里补充上一节的三分量对齐逻辑）。

**`models/qwen3.py` 的改动**：仅把 `QKVParallelLinear(...)` 等直接构造，替换为一个工厂函数：

```python
def make_column_linear(vllm_config, *args, **kwargs):
    if vllm_config.quantization == "awq":
        return AWQColumnParallelLinear(*args, group_size=vllm_config.awq_group_size, **kwargs)
    return ColumnParallelLinear(*args, **kwargs)
```

`Qwen3Attention`/`Qwen3MLP` 里把 `QKVParallelLinear(...)`/`MergedColumnParallelLinear(...)`/`RowParallelLinear(...)` 的直接调用改为等价的工厂函数调用，**其余模型结构代码零改动**。

### 4.5 计算 kernel 路线（对应 `awq_kernel` 配置）

**Phase A（正确性优先）：`awq_kernel = "torch"`**

```python
def awq_gemm_torch(x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    # 1) 反量化整块权重为 fp16：[in_features, out_features]
    w_fp16 = dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor)
    # 2) 复用现有 F.linear
    return F.linear(x, w_fp16.t().contiguous(), bias)
```

- `dequantize_awq_weight` 用 Triton 或纯 Torch 位运算实现"解包 int32→8×int4 + 按 group 广播 scale/zero_point 还原 fp16"。
- 优点：实现简单、正确性容易验证（可直接跟 AutoAWQ 官方推理结果做数值对比）；缺点：每次 forward 都要现场还原一份完整 FP16 权重，**没有省显存带宽的收益**（本质上只省了"静态存储"的显存，没省"计算时的带宽"），性能不会比 FP16 好，甚至因为多了一次反量化开销会更慢。
- **定位**：仅作为"正确性基线 / 显存占用验证"用途，不是最终交付形态。

**Phase B（性能优先）：`awq_kernel = "triton"`，Fused Dequant+GEMM**

参考 AutoAWQ Triton kernel 与 arXiv:2402.00025 的融合思路，写一个 Triton kernel `awq_gemm_kernel`：

- 分块（tile）计算 `Y[M,N] = X[M,K] @ W[K,N]`，在从全局内存加载 `W` 的分块（`qweight` tile）时，**在寄存器/共享内存里现场解包 int4 + 乘 scale + 减 zero_point 得到 fp16 tile**，直接参与本次矩阵乘的累加，不写回一份完整反量化权重到显存。
- 这样真正利用了"读取的是 1/4~1/8 体积的 INT4 权重"的带宽优势，是 decode 阶段加速的关键。
- 与本项目现有 `store_kvcache_kernel` 一样用 Triton 编写，不引入新的编译依赖，符合项目"轻量、可读"的定位。
- **落地建议**：先支持 `M`（batch×seq）较小（decode 场景，GEMV 语义）的分支做重点优化，再逐步补齐 prefill 阶段的大 `M` GEMM 分支（可以先复用 Phase A 的 torch 路径应付 prefill，因为 prefill 是 compute-bound，量化收益本来就小，优先级低于 decode）。

**Phase C（可选兜底）**：留出 `awq_kernel = "ext"` 分支，允许对接社区预编译 kernel（`autoawq-kernels` 或 vLLM 的 Marlin kernel）作为可选性能后端，供后续需要极致性能时按需接入，不作为默认依赖。

### 4.6 Input scale 的处理：AutoAWQ 已自动融合，推理路径无需关心

**问题背景**：AWQ 的等价变换是 `Y = (X/s) @ (W*s)`。量化时 `W' = W*s` 已经被打包进 `qweight/scales`，但推理时输入 `X` 还需要除以 `s` 才能保持数学等价。

**AutoAWQ 的处理方式（方案 B：融合进前一层 RMSNorm weight）**：

AutoAWQ 在**离线量化时**自动把 `1/s` 融合进前一层 RMSNorm 的 weight `γ`（替换为 `γ/s`）。这是 AutoAWQ 标准流程的一部分。

因此**推理路径完全无感**：
- checkpoint 里的 RMSNorm weight **已经是 `γ/s`**（AutoAWQ 量好并存盘的）
- safetensors 里 RMSNorm 的 weight 名不变（如 `model.layers.0.input_layernorm.weight`），但**值**是修改后的 `γ/s`
- RMSNorm 的 forward 代码**不需要任何改动**（`x * (γ/s)` 自然输出 `X/s`）
- AWQLinear 的 forward **不需要额外除法**（`x` 已经是 `X/s`）
- checkpoint 里**不额外存 `input_scales`** tensor

**验证方式**：加载 AutoAWQ 生成的 checkpoint 后，对比 RMSNorm weight 与原始 FP16 模型的 RMSNorm weight，应该不同（因为融合了 `1/s`）。

**特殊情况**：

| 场景 | 说明 |
|---|---|
| Qwen3 所有 Linear 前面都接 RMSNorm | AutoAWQ 覆盖完整，无遗漏 |
| TP 切分 | RMSNorm 是 replicated（非 sharded），`γ/s` 全量复制到每个 rank，无需特殊处理 |
| `lm_head`（不量化） | 不做 scale 融合，保持原始 weight |

### 4.7 与现有子系统的交互确认

- **TP（张量并行）**：AWQ Linear 的 shard 逻辑与现有 FP16 Linear 保持同构（同样是 `ColumnParallelLinear`/`RowParallelLinear` 的语义），差异仅在于要按 `pack_factor`/`group_size` 对齐，见 4.3/4.4。`RowParallelLinear` 的 `all_reduce` 逻辑不变。
- **CUDA Graph（`capture_cudagraph`）**：量化 GEMM kernel（无论 Phase A 的 `F.linear` 还是 Phase B 的 Triton kernel）都是标准的、shape 固定的 kernel 调用，**可以被 CUDA Graph 正常捕获**，不需要像"KV 压缩 step"那样退化到 eager 模式。
- **窗口化 + 周期性 KV 压缩**：压缩机制操作对象是 `k_cache/v_cache`（激活值），与"权重怎么存储/怎么做矩阵乘"完全解耦，**无需任何改动**，两个特性可以同时开启。
- **`allocate_kv_cache` 显存预算**：量化后模型权重显存占用下降（Qwen3-8B 约从 16GB(FP16) 降到 ~4.5GB(INT4)），`config.gpu_memory_utilization` 换算出的可用显存里，权重占用变小，`num_kvcache_blocks` 计算逻辑不用改代码，但实际可分配的 KV cache block 数量会自然变多——这是 AWQ 给本项目带来的"意外收益"：可以支撑更大 `max_num_seqs`/更长上下文。

### 4.8 量化 checkpoint 的获取

本项目**不实现量化算法本身**（校准、scale 搜索、int4 打包），而是使用 AutoAWQ 离线量化生成 checkpoint，这与 vLLM 的做法一致：

```bash
# 离线量化（一次性，在 nano-kvllm 之外完成）
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained('/path/to/Qwen3-1.7B')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4})
model.save_quantized('/path/to/Qwen3-1.7B-AWQ')
"
```

产出的 checkpoint 目录包含：
- `*.safetensors`：包含 `qweight/qzeros/scales`（量化 Linear）+ 普通 weight（RMSNorm/embedding 等）
- `quantize_config.json`：量化元数据（bits/group_size/zero_point）
- tokenizer 等非权重文件

**AutoAWQ 在量化时已自动完成 input scale 的融合**（见 4.6 节），checkpoint 里的 RMSNorm weight 已经是 `γ/s`，推理路径无需任何额外处理。

---

## 5. 分阶段实施路线图

| 阶段 | 目标 | 交付物 | 验收标准 |
|---|---|---|---|
| **Phase 0：快速拿到测试 checkpoint** | 用 AutoAWQ 库离线量化一个小模型（如 Qwen3-0.6B/1.7B），验证推理链路的"标准答案" | 量化脚本 + 测试 checkpoint | 能被 AutoAWQ 官方推理库正常加载生成合理输出 |
| **Phase 1：Config + Loader** | `config.quantization="awq"` 生效，能正确解析 `quantize_config.json`，`load_model` 能把 `qweight/qzeros/scales` 灌进新 Parameter（含 QKV/gate_up 融合的对齐逻辑） | `config.py`、`loader.py` 改动 | 单测：加载后各 Parameter 的 shape/dtype 与 AutoAWQ 官方一致；TP=2/4 时各 rank shard 后 shape 正确 |
| **Phase 2：AWQLinear（torch 反量化）** | 新增 `AWQLinearBase` 及派生类，`qwen3.py` 接入工厂函数，端到端跑通 | `layers/awq_linear.py`、`models/qwen3.py` 少量改动 | 同一 prompt 下，AWQ 路径与 AutoAWQ 官方推理的 logits/生成文本高度接近；`enforce_eager=True` 下可正常生成 |
| **Phase 3：Triton Fused Kernel** | 实现 `awq_gemm_kernel`（decode 场景优先），替换 Phase 2 的性能瓶颈路径 | `layers/awq_triton_kernel.py` | decode 吞吐 / 显存占用对比 benchmark，验证相比 FP16：显存占用下降 ≥60%，decode 吞吐不低于 FP16 |
| **Phase 4：CUDA Graph 兼容 + TP 回归** | 确保 `enforce_eager=False`（图模式）下量化路径正常，TP=2/4/8 全面测试 | 修复兼容性问题 | 图模式下量化推理正确且有性能收益；TP 各配置下生成结果与单卡一致 |
| **Phase 5：与 KV 压缩联合验证** | 同时开启 `quantization="awq"` 与 `kv_compress_enabled=True`，跑完整回归 | 联合测试报告 | 两个特性同时开启时生成质量、吞吐均正常，无相互干扰 |
| **Phase 6（可选）：性能兜底后端** | 按需接入社区预编译 kernel（如 Marlin）作为可选高性能后端 | `awq_kernel="ext"` 适配层 | 需要极致性能时按需启用，不影响默认路径稳定性 |

---

## 6. 验证与验收方法

1. **正确性验证**：
   - 数值对比：固定输入，比较 `nano-kvllm` 量化路径与 AutoAWQ 官方推理的 logits（如 top-1/top-5 一致率、KL 散度）。
   - 端到端质量：在小型评测集（如 GSM8K / MMLU 子集）上对比 FP16 baseline 与 AWQ 版本的准确率差异，应在可接受范围内（社区经验：4bit AWQ 相比 FP16 精度损失通常 <1~2 个百分点）。
2. **性能验证**（复用/扩展现有 `bench.py`）：
   - 显存占用：加载后 `torch.cuda.memory_stats()` 对比权重部分显存占用。
   - 吞吐：不同 batch size（1 / 32 / 256）下的 decode tok/s 对比，重点关注低并发场景（AWQ 收益最明显的区间），同时如实记录高并发下可能出现的"收益收窄甚至持平"现象（第 2.1 节已提示这是社区公认的边界情况）。
3. **回归验证**：确保关闭量化（`quantization=None`）时，现有全部功能（TP、CUDA Graph、KV 压缩）行为与改造前完全一致（这部分改动应该是纯增量、无侵入）。

---

## 7. 风险与开放问题

| 风险/问题 | 说明 | 应对思路 |
|---|---|---|
| 融合层（QKV/gate_up）与量化打包的维度对齐 | `q_proj/k_proj/v_proj` 各自 `output_size` 必须是 `pack_factor × tp_size` 的整数倍，否则无法均匀切分 | 加载期显式断言 + 报错提示，必要时要求用户使用 `tp_size` 能整除的量化配置重新量化 |
| Triton fused kernel 开发/调优成本 | Phase 3 是本方案里工程量最大、最考验 Triton 调优能力的部分 | 分阶段交付，Phase 1/2 先保证功能正确可用（哪怕性能不如 FP16），Phase 3 再迭代优化，不阻塞整体可用性 |
| 高并发下收益有限 | vLLM/AutoAWQ 社区经验：大 batch 下 AWQ 相对 FP16 的吞吐收益会明显收窄 | 在文档/README 中如实说明适用场景（低并发/长上下文/显存受限场景收益最大），避免过度承诺 |
| CUDA Graph + Triton kernel 的兼容性 | 需确认自定义 Triton kernel 在 `torch.cuda.graph` capture 下行为正确（无动态 shape/host-side 分支） | Phase 4 专项测试；若发现不兼容，量化 decode step 可参照现有"压缩 step 退化到 eager"的模式处理 |
| AutoAWQ checkpoint 的版本兼容性 | 不同 AutoAWQ 版本的打包格式（交织顺序、qzeros +1 偏移）可能略有差异 | Phase 0 用明确的 AutoAWQ 版本生成 checkpoint，并在 `dequantize` 实现后做数值对比验证；如遇格式差异，在 `dequantize` 里做版本适配 |

---

## 8. 结论

AWQ 是一种**激活感知、无需反向传播、按 group 做 INT4 非对称量化**的 weight-only 量化方法，核心是用"等价的 per-channel scaling"把混合精度问题转化为纯 INT4 量化问题。业界（AutoAWQ/vLLM）已有成熟的打包格式（`qweight/qzeros/scales`）和两类 kernel 范式（GEMM 面向大 batch，GEMV 面向低并发），本项目风格更贴近**自研 Triton fused kernel**路线。

本项目**不实现量化算法本身**，而是使用 AutoAWQ 离线生成的 checkpoint，这与 vLLM 的做法一致——vLLM 同样只做量化推理，不做量化过程。接入 `nano-kvllm` 的改造集中在 **`config.py`（开关与元数据）→ `loader.py`（三分量权重加载与 TP 切分对齐）→ `layers/linear.py`（新增 AWQLinear 系列 + 工厂分发）** 三处，`models/qwen3.py` 只需极少改动，**与窗口化+周期性 KV 压缩机制完全正交、无需改动**。建议按"先正确性（torch 反量化）、后性能（Triton fused kernel）"分阶段推进，优先保证低并发/长上下文场景下的显存与吞吐收益。
