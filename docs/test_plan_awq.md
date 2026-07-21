# AWQ 量化测试方案

> 版本：draft v1
> 作者：nano-kvllm
> 适用代码基线：nano-kvllm v0.2.0
> 关联脚本：`quantize_awq.py`（量化产物生成）、`bench_awq.py`（性能基准）

---

## 1. 目标与指标

### 1.1 目标
1. **正确性**：验证 AWQ 量化模型能在 nano-kvllm 中正确加载并产出通顺输出。
2. **精度**：量化后与 FP16 baseline 相比，输出质量下降在可接受范围内（token 重合率 / perplexity / 下游任务分数）。
3. **加速比**：定量对比 AWQ vs FP16 在 **TTFT / TPOT / 吞吐 / 峰值显存** 上的差异。
4. **kernel 选型**：对比 `awq_kernel="torch"`（反量化+F.linear）与 `awq_kernel="triton"`（融合 GEMM）的性能，明确各场景推荐值。

### 1.2 指标定义

| 指标 | 定义 | 测量方式（本框架） | 单位 |
|---|---|---|---|
| **TTFT** | Time To First Token；请求提交到第一个 token 返回的时延 | batch=1，第一次 `step()`（prefill）耗时 | ms |
| **TPOT** | Time Per Output Token；decode 阶段单 token 平均时延 | batch=1，每个 decode step 只产 1 token，取 mean(decode step 耗时) | ms/tok |
| **decode tok/s** | 1000 / TPOT | 派生 | tok/s |
| **E2E Latency** | 端到端；请求提交到完成 | 从 add_request 到 seq.is_finished 的墙钟 | ms |
| **Gen Throughput** | 大 batch 吞吐 | 总输出 token / 总时间（`generate()` 全流程） | tok/s |
| **Peak Memory** | 峰值显存 | `torch.cuda.max_memory_allocated()` | GB |
| **加速比 (Speedup)** | FP16 指标 / AWQ 指标（延迟类）或 AWQ / FP16（吞吐类） | 派生 | × |
| **精度保真** | AWQ 输出 top-1 与 FP16 输出 top-1 一致率 | 相同 seed + 相同 prompt，逐 token 比对 | % |

### 1.3 预期方向
- **TPOT**：AWQ 应显著优于 FP16（decode 是 memory-bound，权重体积 ~1/4，理论 3× 上限，triton kernel 实测 1.5~2.5×）。
- **TTFT**：AWQ 优势较小甚至可能变慢（prefill 是 compute-bound，反量化开销可能吃掉带宽收益）。torch kernel 尤其明显。
- **吞吐**：大 batch 下 decode 占比升高，AWQ 优势放大。
- **显存**：模型权重节省 ~65~75%（4bit + group scale/zeros），但 KV cache 不变；因此**总显存节省比例取决于 model_size vs kv_cache 占比**。

---

## 2. 前置准备

### 2.1 依赖

```bash
pip install autoawq                # 生成量化权重
pip install lm-eval                # （可选）下游任务评测
```

CUDA / PyTorch / triton 与现有环境保持一致。

### 2.2 兼容性要点（重要）

nano-kvllm 的 AWQ 加载有 3 个隐性约束，量化脚本必须处理：

1. **`version="GEMM"`**：`awq_gemm.py::dequantize_awq_weight` 的解包顺序（`uint8` 拆高低位 + 交织）与 AutoAWQ 的 GEMM pack 格式一一对应。用 GEMV / Marlin 会解出错误权重。
2. **跳过 `lm_head` / embedding**：Qwen3 模型只把 `qkv_proj / o_proj / gate_up_proj / down_proj` 走 AWQ 层，embed_tokens 和 lm_head 是普通 FP16。AutoAWQ 默认就跳过 lm_head，需要确认没有强行量化。
3. **`quantize_config.json` 键名对齐**：`config.py` 读取 `bits / group_size / zero_point`，而 AutoAWQ 默认导出 `w_bit / q_group_size`。**必须手动写出兼容版本的 quantize_config.json**（已在 `quantize_awq.py` 中处理）。

### 2.3 量化产物生成

```bash
python quantize_awq.py \
    --model ~/huggingface/Qwen3-8B \
    --out   ~/huggingface/Qwen3-8B-AWQ \
    --bits 4 --group_size 128
```

产物目录应包含：
- `model.safetensors*`（量化后权重）
- `config.json`（模型结构，来自 AutoAWQ）
- `quantize_config.json`（**nano-kvllm 兼容版**，键名 `bits/group_size/zero_point`）
- `tokenizer.*`

### 2.4 校准数据集
AutoAWQ 默认用 `pileval` 或 `c4` 的小样本。若模型对话/数学场景专用，建议：
- 通用：默认即可。
- 数学 / 长思考：从 Math500 抽 ~256 条 prompt 作为校准语料（提升下游任务保真）。

---

## 3. 正确性验证（P0，必过）

### 3.1 冒烟测试
用 `example.py` 改写：加载 AWQ 模型，喂一个 Math500 prompt，验证输出可读、含 `\boxed{...}`。

命令：
```bash
python example.py  # 修改 path 为 AWQ 目录，config 加 quantization="awq"
```

**通过标准**：
- 无加载报错（weight shape 匹配）；
- 输出不是重复 token / 乱码；
- 数学问题给出合理的 CoT + 最终答案。

失败诊断：
- 权重 shape mismatch → 检查 `pack_factor` (32/bits)、`group_size`；
- 输出乱码 → 检查 zero_point 是否 `-1` 处理；`version` 是否 GEMM；
- 数值全 NaN → 检查 `dequantize_awq_weight` 的 int4 unpack；

### 3.2 kernel 一致性
`awq_kernel="torch"` 与 `awq_kernel="triton"` 输出的 top-1 token 应完全一致（同 seed 同 prompt），逐 token 比对：

**通过标准**：token 序列一致率 100%（首 200 token）。若不一致 → triton kernel 存在数值 bug，优先修 kernel。

---

## 4. 精度保真验证（P1）

### 4.1 Top-1 一致率
- 从 Math500 / GSM8K 各取 20 条 prompt；
- 同 seed，`temperature=0.6`（框架不允许 0），`max_tokens=256`；
- 对每条 prompt，比较 FP16 vs AWQ 逐 token 的 top-1；
- 汇报 **前 K token 累计一致率**（K=20, 50, 100, 200）。

**通过标准（经验值，实际可根据业务调整）**：
- K=20：≥ 90%
- K=100：≥ 70%
- K=200：≥ 55%

### 4.2 Perplexity（可选，更严格）
- WikiText-2 test split，滑窗 PPL；
- FP16 vs AWQ 的 PPL 相对差应 ≤ 5%（4bit AWQ 的典型退化区间）。

### 4.3 下游任务（可选）
- 用 lm-eval-harness 跑 `gsm8k` / `hellaswag` / `mmlu` 的子集；
- 记录 acc / acc_norm；相对退化 ≤ 2 个百分点视为可接受。

---

## 5. 性能基准（核心）

### 5.1 测试脚本设计要点

已提供 `bench_awq.py`，关键设计：
- **单进程只测一个配置**：TP 会 `init_process_group`，不可在同进程切换模型；baseline 与 AWQ 分开跑，脚本外部对比。
- **控制变量**：
  - `kv_compress_enabled=False` 强制关闭（排除压缩逻辑对 decode 计时污染）；
  - `ignore_eos=True` + 固定 `max_tokens`（保证 decode 步数完全相同）；
  - 相同 seed（`random.seed(0)`）；
  - warmup 2 次丢弃，5 次取中位数。
- **TTFT/TPOT 分离**：手动驱动 `engine.step()` 循环，`num_tokens > 0` 判定 prefill step。

### 5.2 测试矩阵

**A. 延迟基准（batch=1）**

| 变量 | 取值 |
|---|---|
| model | Qwen3-0.6B / Qwen3-8B（小模型验流程，大模型看真实场景） |
| quant | none / awq(torch) / awq(triton) |
| input_len | 128, 512, 2048 |
| output_len | 128, 512 |
| enforce_eager | on, off |
| TP | 1（8B 可加测 TP=2）|

= 2 × 3 × 3 × 2 × 2 × 1 = 72 组，可按 P2/P3 优先级筛选（先跑 8B, output=256, eager on/off 各一）。

**B. 吞吐基准（batch>1）**

| 变量 | 取值 |
|---|---|
| batch | 1, 8, 32, 64 |
| input_len | 512 |
| output_len | 256 |
| quant | none / awq(triton) |
| enforce_eager | off（吞吐场景必开图） |

### 5.3 执行示例

```bash
# baseline FP16
python bench_awq.py --model ~/huggingface/Qwen3-8B --quant none \
    --enforce_eager --input_len 512 --output_len 256 --throughput_bs 32 \
    2>&1 | tee logs/fp16_eager_bs32.log

python bench_awq.py --model ~/huggingface/Qwen3-8B --quant none \
    --input_len 512 --output_len 256 --throughput_bs 32 \
    2>&1 | tee logs/fp16_graph_bs32.log

# AWQ torch kernel
python bench_awq.py --model ~/huggingface/Qwen3-8B-AWQ --quant awq --awq_kernel torch \
    --enforce_eager --input_len 512 --output_len 256 --throughput_bs 32 \
    2>&1 | tee logs/awq_torch_eager_bs32.log

# AWQ triton kernel
python bench_awq.py --model ~/huggingface/Qwen3-8B-AWQ --quant awq --awq_kernel triton \
    --input_len 512 --output_len 256 --throughput_bs 32 \
    2>&1 | tee logs/awq_triton_graph_bs32.log
```

---

## 6. 结果记录与分析

### 6.1 主表模板

**延迟对比（batch=1, input=512, output=256, TP=1, enforce_eager=False）**

| 配置 | TTFT (ms) | TPOT (ms/tok) | decode (tok/s) | E2E (ms) | Peak Mem (GB) | 精度(K=100 top1一致率) |
|---|---|---|---|---|---|---|
| FP16 | 基准 | 基准 | 基准 | 基准 | 基准 | 100% |
| AWQ-torch | | | | | | |
| AWQ-triton | | | | | | |
| **Speedup vs FP16** | TTFT: | TPOT: | decode: | E2E: | 显存节省%: | — |

**吞吐对比（input=512, output=256, TP=1）**

| Batch | FP16 tok/s | AWQ-triton tok/s | Speedup | FP16 Peak(GB) | AWQ Peak(GB) | 显存节省 |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 8 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |

### 6.2 曲线图（推荐）
- **input_len → TTFT** 折线，三条曲线对比 FP16 / AWQ-torch / AWQ-triton；
- **batch → Throughput** 折线，观察 AWQ 优势随 batch 变化；
- **batch → Peak Memory** 柱状，验证 KV cache 占比。

### 6.3 分析维度
1. **TPOT 加速比**：应最显著（memory-bound）。分析 triton vs torch 差距。
2. **TTFT 变化**：若 AWQ-torch 比 FP16 慢，说明反量化开销未被抵消；triton 应有正加速。
3. **图模式增益**：对同一 quant，比 eager vs graph 的 TPOT，量化后 launch 开销占比下降，图收益可能变小。
4. **大 batch 显存**：AWQ 释放的显存可拿去装更大 KV cache（对应 `num_kvcache_blocks` 变多），间接提升可服务并发数——建议单独测一个"最大可承载 batch"指标。

---

## 7. 消融与专题

### 7.1 Kernel 深挖（triton vs torch）
在 prefill / decode 两个阶段分别做**单层前向 micro-benchmark**（脱离全模型）：
- 输入 `x ∈ [tokens, hidden]`，固定 hidden=4096，扫 tokens ∈ {1, 8, 32, 128, 512, 2048}；
- 记录 `AWQGemmTorch.__call__` vs `AWQGemmTriton.__call__` 各自延迟；
- 对比 FP16 `F.linear`（同 shape）；
- 输出：**三条曲线 vs tokens**。

期望结论：
- tokens=1 附近：triton 应 3~5× 优于 torch，triton ≥ FP16；
- tokens 大（≥256）：torch 差距缩小；triton 相对 FP16 优势也缩小。

### 7.2 Group size 消融（可选）
`group_size ∈ {64, 128}` 分别量化一份，测：
- 精度（PPL）差异；
- 性能差异（scale/zeros 表变小对 GEMM 影响）。

### 7.3 TP 影响
- Qwen3-8B 用 TP=2 vs TP=1；
- 验证 AWQ 权重切分逻辑正确（各 rank 输出应对齐）；
- 记录 TPOT 是否随 TP 线性下降。

---

## 8. 已知风险与应对

| 风险 | 检测方式 | 应对 |
|---|---|---|
| AutoAWQ 版本变化导致 pack 格式改变 | 冒烟测试输出乱码 | 版本锁定；对比 `version` 字段；必要时调整 `dequantize_awq_weight` |
| `zero_point +1 → -1` 处理错误 | 精度差异极大（PPL 翻倍） | 检查 `int4_zeros -= 1`；用少量样本回归 |
| triton kernel 数值精度问题 | §3.2 一致性测试失败 | 优先修 kernel；不影响时用 torch 作 fallback |
| TP 切分对 `group_size` 不整除 | 权重加载报 shape assert | 换 group_size 或切分维度 |
| KV cache 未量化 → 大 batch 显存瓶颈仍在 | Peak Memory 测量表 | 记录，作为后续 KV quant 的动机 |
| CUDA graph 桶未覆盖到某 batch → 走 eager | log 里看到 non-graph path | 图桶列表按测试 batch 配置 |

---

## 9. 交付清单

- [ ] `quantize_awq.py`（已提供）
- [ ] `bench_awq.py`（已提供）
- [ ] `~/huggingface/Qwen3-8B-AWQ/`（量化产物 + 兼容 quantize_config.json）
- [ ] `logs/` 目录下 baseline + AWQ(torch) + AWQ(triton) × {eager, graph} × batch 组合的 log
- [ ] `report_awq.md`：填好的主表 + 曲线图 + 分析结论
- [ ] （可选）`bench_awq_micro.py`：单层 GEMM micro-benchmark
- [ ] （可选）`eval_awq.py`：lm-eval 下游任务脚本

---

## 10. 执行 checklist

**Day 1**
1. 安装 autoawq，生成 Qwen3-0.6B AWQ 权重（小模型走通流程）。
2. 冒烟测试 + torch/triton 一致性。
3. 生成 Qwen3-8B AWQ 权重。

**Day 2**
1. 精度验证（§4.1，K=100 一致率）。
2. 延迟主矩阵（batch=1，input=512，output=256，三种量化配置 × eager/graph）。

**Day 3**
1. 吞吐矩阵（batch=1/8/32/64）。
2. 峰值显存记录。
3. 填主表，画曲线图。

**Day 4（可选）**
1. Micro-benchmark（§7.1）。
2. TP=2（§7.3）。
3. 撰写 `report_awq.md`。
