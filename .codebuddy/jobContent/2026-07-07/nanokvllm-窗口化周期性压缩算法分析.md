# nano-kvllm：窗口化 + 周期性 KV Cache 压缩机制代码分析

> 分析对象：`nanokvllm/` 目录（v0.2.0），对比基线为原始 `nano-vllm`。
> 目标：讲清楚"窗口化压缩 + 周期性触发"这一套 KV Cache 动态压缩机制的设计动机、端到端数据流、关键代码实现与边界处理。

---

## 1. 背景：为什么不用"阈值触发压缩"

`nano-kvllm` 早期版本（v0.1.0/v0.1.5，逻辑保留在 `KvChat/` 中）采用的是业界常见的**阈值触发（threshold-triggered）**压缩：单条序列 KV 长度达到 `S` 就压缩到 `R`。这种方式在单用户/Agent 场景够用，但在高并发批量推理场景有两个问题（README 中明确提到）：

1. **可能压到 system prompt**：压缩基于"整段历史 KV"选择保留 token，可能把系统提示词相关的 KV 剪掉，影响生成质量。
2. **高 batch 下拖累吞吐**：几乎每个请求在每个 decode step 都可能各自触发一次压缩，压缩开销叠加，反而降低吞吐。

`nanokvllm` v0.2.0 针对这两点提出了**窗口化（window-based）+ 周期性（periodic）**的压缩机制，即本次分析的核心。

---

## 2. 核心思路

- **周期性触发**：全局维护一个 decode step 计数器，每隔固定步数（`kv_compress_period`，默认 1024）才触发一次压缩，而不是"谁先达到阈值谁先压"。避免了压缩事件在 batch 内到处发生。
- **Top-K 候选**：触发时，从当前 batch 里挑选最多 `kv_compress_topk`（默认 20）条满足条件的序列参与本次压缩，而不是全员压缩，控制单步压缩开销。
- **窗口化（保护区）**：每条被选中序列的**最后 `kv_compress_window_blocks` 个完整 block**（默认 4 个 block）作为压缩窗口——**只压这个窗口内的 KV**，窗口外（更早）的 KV 保持不变、且**不会被重复压缩**（因为窗口外的内容早已是之前压缩后的结果或原始 prompt）。
- **压缩算法**：窗口内用 SnapKV 思路——用**当前 decode step 的最新 token 的 query** 对窗口内的 K 做注意力打分，选出重要的 KV，保留 `kv_compress_keep_blocks*block_size + kv_compress_keep_extra_tokens` 个 token。
- **物理内存搬迁（compaction）**：保留的 KV 在物理 slot 维度做 in-place 紧凑搬迁，同时更新 `context_lens`；压缩事件在最后一层记录，回传给 scheduler 做 `block_table` 截断和物理 block 释放。

---

## 3. 端到端数据流

```text
ModelRunner.prepare_decode(seqs)                     # 每个 decode step 之前
  ├─ decode_step_counter += 1
  ├─ is_compress_step = (counter % kv_compress_period == 0)
  ├─ 若是压缩步：遍历 seqs，筛出满足"窗口已攒够"的候选，取前 topk 个
  │     └─ 候选条件：seq.tail_uncompressed_len >= window_tokens
  │                 且 full_blocks >= window_blocks
  │                 且 当前 tail 不处于"整块差1"的边界态
  └─ 写入 Context: is_compress_step / compress_selected_batch_indices
                   / compress_base_context_lens（本步压缩前的 context_lens 快照）

Qwen3ForCausalLM.forward → 逐层 DecoderLayer → Qwen3Attention.forward(Layer)
  └─ Attention.forward(q, k, v, Layer)
        ├─ store_kvcache(...)                        # 先写入本 step 新 token 的 KV
        ├─ if is_compress_step and 有被选中序列:
        │     MyCompressCompact(q_current=q, ..., layer_id=Layer)
        │         ├─ 定位每条被选序列尾部 window_blocks 个整块 + 可能的半满尾块
        │         ├─ gather 出窗口内 K/V
        │         ├─ SnapKV(q, k_window, v_window) → keep_idx
        │         ├─ 物理 slot 级别 compact（index_select + index_copy_）
        │         ├─ 更新 context.context_lens[seq_idxs]
        │         └─ 若 layer_id 是最后一层：记录 compression_events
        └─ flash_attn_with_kvcache(...)               # 用压缩后的 context_lens/cache 做注意力

ModelRunner.run() → 收集 compression_events（仅 rank0）→ 返回给上层

LLMEngine.step() → Scheduler.postprocess(seqs, token_ids, compression_events)
  ├─ 按 batch_index 去重（每条序列只应用最后一次事件）
  ├─ BlockManager.truncate_blocks(seq, keep_blocks)    # 释放窗口压缩后多余的物理 block
  ├─ seq.num_tokens = new_context_len                  # 物理缓存长度更新
  └─ seq.tail_uncompressed_len = 0                      # 重新开始累计"未压缩窗口"
```

---

## 4. 关键代码逐文件解析

### 4.1 `config.py` —— 压缩相关超参

```python
kv_compress_enabled: bool = True        # 是否开启压缩（仅作用于 decode，不压 prefill）
kv_compress_period: int = 1024          # 全局周期：每多少个 decode step 触发一次压缩
kv_compress_topk: int = 20              # 每次压缩最多选取多少条候选序列
kv_compress_window_blocks: int = 4      # 压缩窗口大小（block 数），窗口内才会被压缩
kv_compress_keep_blocks: int = 2        # 压缩后窗口内保留多少个完整 block
kv_compress_keep_extra_tokens: int = 1  # 额外保留 token 数（配合 bos + latest token 语义）
```

`kv_compress_keep_blocks*block_size + kv_compress_keep_extra_tokens` = 压缩后窗口保留的 token 数（对应 README 中的 "50% 压缩率" 等参数组合）。

### 4.2 `engine/sequence.py` —— 逻辑长度与物理长度解耦

新增字段：

- `rope_pos`：单调递增，仅用于 RoPE 位置编码，**压缩不影响它**（否则位置编码会错乱）。
- `generated_completion_tokens`：逻辑生成 token 数，用于判断是否达到 `max_tokens`。
- `tail_uncompressed_len`：**自上次压缩以来新增的 token 数**，用于判断"窗口是否已经攒够，可以参与本轮压缩候选"。压缩发生后被重置为 0（见 `scheduler.postprocess`）。
- `num_tokens`：物理 KV cache 有效长度（会在压缩后变小）。

`append_token()` 每次同时递增 `num_tokens / generated_completion_tokens / rope_pos / tail_uncompressed_len`，`__getstate__/__setstate__` 也做了相应的序列化适配（用于 TP 场景下跨进程传递序列状态）。

### 4.3 `engine/model_runner.py::prepare_decode()` —— 周期触发 + Top-K 候选筛选

这是"何时压、压哪些序列"的决策点，在**进入模型前**、每个 decode step 统一计算一次，避免在每一层 Attention 里重复判断（README 提到这带来 5%~10% TPS 提升）。

```python
self.decode_step_counter += 1
is_compress_step = (self.decode_step_counter % self.config.kv_compress_period == 0)

if is_compress_step:
    candidates = []
    for i, seq in enumerate(seqs):
        current_context_len = len(seq)
        full_blocks = current_context_len // B
        tail_len = current_context_len % B
        if tail_len == B - 1:
            continue                                   # 边界态：马上要整块了，跳过避免冲突
        elif seq.tail_uncompressed_len >= window_tokens and full_blocks >= window_blocks:
            candidates.append((i, seq.seq_id))
    selected = candidates[:topk]
```

要点：
- 候选条件里的 `seq.tail_uncompressed_len >= window_tokens` 保证**只有攒够了一整个窗口的"新鲜"token 才会被再次压缩**——这正是"窗口化"防止重复压缩已压缩内容的关键判据。
- `context.compress_base_context_lens = context_lens.clone()`：**在任何层修改 `context_lens` 之前**，先快照一份原始长度，供 `MyCompressCompact` 内部计算 block 定位时使用（因为多层共享同一个 `context_lens`，第一层压缩后会改变它，后续层不能再用变化后的值定位窗口）。

### 4.4 `layers/attention.py::Attention.forward()` —— 压缩插入点

```python
store_kvcache(...)                 # 先写入本 step 新 token
if (not is_prefill) and kv_compress_enabled and is_compress_step and selected_batch_indices:
    MyCompressCompact(q_current=q, k_cache=k_cache, v_cache=v_cache, layer_id=Layer, ...)
flash_attn_with_kvcache(...)       # 再做注意力（此时 context_lens 已是压缩后的值）
```

即"先写新 KV，再压缩，最后做 attention"，保证注意力计算时看到的是**压缩后**且**包含最新 token**的 cache。压缩只在 decode 阶段生效（`context.is_prefill` 为 False 时）。

### 4.5 `layers/compress_utils.py::MyCompressCompact()` —— 核心压缩+搬迁逻辑

分为几个阶段：

1. **定位压缩窗口**（`get_tail_window_and_tail_slots`）
   - 用 `compress_base_context_lens` 计算每条选中序列尾部有多少个完整 block（`full_blocks`）、是否有半满尾块（`tail_len`）。
   - 取最后 `window_blocks` 个完整 block 对应的物理 slot（`window_src_slots`），以及半满尾块的物理 block id（`tail_block_ids`，若没有则为 -1）。
   - 这一步全部向量化（`torch.gather`/`torch.where`），一次处理 `m` 条被选序列，没有 Python 级 for 循环遍历 token。

2. **取出窗口内 K/V**（`gather_kv_by_slots`）：把 `[num_blocks, block_size, H, D]` 的物理 cache 按 `window_src_slots` 索引取出，reshape 成 `[m, H, window_tokens, D]`。

3. **调用压缩算法** `SnapKV(q_sub, k_sub, v_sub, num_keep=..., window=1)`（见 4.6），得到窗口内要保留的相对下标 `keep_idx`（形状 `[m, keep_tokens]`）。

4. **物理搬迁（compaction）**：
   - `src_keep = gather(window_src_slots, keep_idx)`：保留 token 的**绝对物理 slot**。
   - `dst_keep = window_src_slots[:, :keep_tokens]`：**目的地**直接用窗口区域最前面的 `keep_tokens` 个 slot（窗口内前压后挪，不需要额外分配新 block）。
   - 若尾部还有半满的 partial block（未参与压缩的最新几个 token），把它**紧跟在 keep 区域后面**一起搬（`dst_tail_start = dst_keep[:, -1] + 1`）。
   - 最终用 `index_select` + `index_copy_` 在扁平化的 `[total_slots, D]` 视图上做搬迁，实现"stable compact"（v0.1.5 之后从 Triton kernel 换成了更稳的纯 Torch 实现）。
   - 更新 `context.context_lens[seq_idxs] = new_context_lens`（`= old_len - window_tokens + keep_tokens`），这样本 step 及之后的 `flash_attn_with_kvcache` 立刻按新长度读取。

5. **仅在最后一层记录压缩事件**（`if layer_id + 1 >= num_layers`）：
   - 因为所有层共享同一份 `k_cache/v_cache/context_lens` 结构且压缩幅度相同，只需要在最后一层算出 `freed_block_ids`（压缩后不再需要的尾部物理 block），打包进 `context.compression_events`，供 `Scheduler.postprocess` 统一处理 `block_table`/内存释放。

### 4.6 `layers/CompressMethod.py::SnapKV()` —— 压缩算法本体

```python
def SnapKV(Q, K, V, num_keep=220, window=5):
    # Q: [B, Hq, window, D]  当前用 window=1，即只用"最新 token 的 query"
    # K: [B, Hk, L, D]       L = window_tokens（压缩窗口内的 K 长度）
```

- 用 `K[:, :, :-window, :]` 排除最后 `window` 个 token（它们本身就会被强制保留，不需要参与打分）。
- 计算注意力分数并对 attention-sink（第 0 个 token）打 `-inf`，避免 softmax 被 sink token 主导。
- 支持 GQA（`Hq != Hk` 时按 `group_size` reshape 做分组注意力）。
- `key_importance` 先在 query-window 维度求和，再在 head 维度求和，得到每个 token 一个总重要性分数，`topk` 选出 `num_keep` 个。
- 最终返回 `[bos_idx, 排序后的 topk_idx, tail_idx(最新 window 个 token)]` 拼接的下标——即**永远保留第一个 token（类似 attention sink / BOS）+ 重要 token + 最新 token**。

这与经典 SnapKV 论文思路一致：用最近若干 query 的注意力分布来估计"哪些历史 KV 对未来预测重要"，是一种**基于注意力得分的稀疏化压缩算法**（不是 window/periodic 本身，而是被嵌入到 window+periodic 框架里的"压缩打分函数"，可替换）。

### 4.7 `engine/scheduler.py::postprocess()` —— 应用压缩结果

```python
for bidx, ev in dedup.items():           # 按 batch_index 去重，只取最后一次事件
    seq = seqs[bidx]
    self.block_manager.truncate_blocks(seq, ev["keep_blocks"])
    seq.num_tokens = ev["new_context_len"]
    seq.tail_uncompressed_len = ev.get("tail_uncompressed_len_after", 0)   # 重新计数
```

### 4.8 `engine/block_manager.py::truncate_blocks()` —— 释放多余物理 block

```python
def truncate_blocks(self, seq, keep_blocks):
    tail = seq.block_table[keep_blocks:]
    for block_id in reversed(tail):
        ...ref_count -= 1; 若为 0 则回收进 free_block_ids...
    seq.block_table = seq.block_table[:keep_blocks]
    seq.num_cached_tokens = min(seq.num_cached_tokens, keep_blocks * block_size)
    if seq.block_table:
        seq.block_table[-1].hash = -1     # 最后一个保留 block 标记为"非满"，避免后续 append 断言失败
```

这是压缩链路上唯一真正"归还物理显存"的地方——前面 `MyCompressCompact` 只是在**已分配的物理 slot 范围内**做搬迁压缩，真正释放 block 让给别的序列复用，是靠这里。

### 4.9 `engine/model_runner.py::run_model()` —— 与 CUDA Graph 的兼容

```python
need_eager_decode = (not is_prefill) and getattr(context, "compress_any", False)
if is_prefill or self.enforce_eager or bs > 512 or need_eager_decode:
    ...走 eager 模式...
else:
    ...replay CUDA graph...
```

设计意图（对应 README v0.1.5 "Graph-mode-aware compression"）：**非压缩 step 走 CUDA Graph 保吞吐，压缩 step 退回 eager 模式**，因为压缩逻辑里有动态 shape/索引操作，天然不适合 graph capture。当前代码里 `compress_any` 始终为 `False`（`prepare_decode` 里显式置为 `False` 且未在压缩逻辑中重新赋值），说明这部分开关在 v0.2.0 里**处于保留但未完全接入**的状态，实际是否退回 eager 主要取决于 `enforce_eager` 配置或 batch size，值得注意。

---

## 5. 关键设计细节 / 边界处理

| 设计点 | 说明 |
|---|---|
| **逻辑位置 vs 物理长度解耦** | `seq.rope_pos`（RoPE 位置，单调递增不受压缩影响）与 `seq.num_tokens`（物理 cache 长度，压缩后变小）分离，避免压缩改变模型对 token 顺序/位置的感知。 |
| **压缩基准长度快照** | `compress_base_context_lens` 在 `prepare_decode` 阶段快照，避免多层共享的 `context_lens` 在前面层被压缩修改后，后面层用"已变化的长度"错误定位窗口。 |
| **仅在最后一层记录事件** | 所有层的压缩幅度一致，`compression_events` 只需要在 `layer_id+1 >= num_layers` 时生成一次，减少重复计算，回传给 CPU 侧统一处理。 |
| **尾部半满 block 处理** | 若序列长度不是 block_size 整数倍，窗口最后可能有个"半满 block"；压缩时把它单独 gather 出来，压缩完后紧跟着 keep 区域重新排布，保证 block_table 语义连续。 |
| **边界态跳过** | `tail_len == B - 1`（下一个 token 就会占满一个 block）时不参与本轮压缩候选，避开压缩与"新 block 分配"时机冲突。 |
| **TP 一致性** | 各 rank 各自压缩自己负责的 heads，但都基于同一份 `context_lens/block_tables`，压缩幅度（`R`/`keep_tokens`）在设计上必须所有 rank 一致，否则会导致 KV 长度在各 rank 间错位。 |
| **前缀哈希复用降级** | 压缩会改变 block 内容，`BlockManager` 的 hash 复用机制在压缩场景下被降级处理（`truncate_blocks` 里直接把最后保留 block 的 `hash` 置为 `-1`），优先保正确性而非前缀复用率。 |
| **压缩不作用于 prefill** | `Attention.forward` 里判断 `not context.is_prefill` 才会调用 `MyCompressCompact`，理由是 decode 是吞吐瓶颈路径，且 prefill 压缩会影响 TTFT。 |

---

## 6. 与 `KvChat`（旧一代压缩方案）的关系

`KvChat/` 目录保留了 v0.1.x 的**阈值触发（S→R）+ `query_window_manager`** 方案（单用户多轮对话场景），与 `nanokvllm` v0.2.0 的**窗口化+周期性**方案是**两代不同设计**，核心差异：

| | KvChat（阈值触发） | nanokvllm v0.2.0（窗口+周期性） |
|---|---|---|
| 触发方式 | 单序列长度达到 `S` 即压缩到 `R` | 全局 decode step 计数，每 `period` 步触发一次 |
| 参与压缩的序列 | 达到阈值的序列（可能全 batch 同时触发） | 触发步内最多 `topk` 条候选序列 |
| Query 来源 | 需要专门的 `query_window_manager` 维护最近 W 个 query 的缓存 | 直接用**当前 decode step 的 query**，无需额外 query cache |
| 适用场景 | 单用户/低并发长对话 | 高并发批量推理，控制压缩开销对吞吐的影响 |

两者共享同一套底层压缩打分算法思路（SnapKV 风格），但调度/触发层完全不同，是本仓库"研究不同压缩落地策略"的两个产物。

---

## 7. 一句话总结

`nanokvllm` v0.2.0 把 KV 压缩从"每条序列各自阈值触发"改造成"全局周期触发 + Top-K 限流 + 仅压缩尾部固定窗口"，用 `context_lens` 快照 + 向量化 slot 搬迁的方式，在保证 PagedAttention 物理内存正确性的前提下，把压缩开销从"处处发生"收敛成"批量、周期性、可控"的事件，从而在高并发长上下文场景下换取更高的解码吞吐（README 实验：Math500 数据集上 2000→2200 tok/s，+10%）。
