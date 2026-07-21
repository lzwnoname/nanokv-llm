# Chunked Prefill + Decode 优先调度 实现方案

> 版本：draft v1
> 作者：nano-kvllm
> 适用代码基线：nano-kvllm v0.2.0
> 状态：设计中，未启动实现

---

## 1. 目标与非目标

### 1.1 目标
1. **调度模型改为混合调度**：一次 `step()` 的 batch 内可以同时包含 decode 序列（每序列 1 个新 token）与 prefill chunk（每序列 N 个新 token）。
2. **decode 优先**：`running` 中已完成 prefill 的序列在每个 step 都能推进 1 个 token，避免被大 prompt 长时间阻塞，稳定 TPOT。
3. **chunked prefill**：waiting 中的新序列按 `chunk_size` 分片进入 batch，让 TTFT 平滑，防止单个长 prompt 独占 GPU。
4. **保留 CUDA graph 收益**：稳态（纯 decode）batch 仍走 graph fast path；混合 batch 走 eager varlen。
5. **KV 压缩暂时禁用**：`kv_compress_enabled=False` 强制关闭，代码保留但不参与本次改造。

### 1.2 非目标（本期不做）
- Prefill/decode 混合场景下的 KV 压缩适配。
- Piecewise CUDA graph（把 non-attn 部分单独捕获成图）。
- Prefix cache 命中率优化。
- 抢占/重算的重写（仅保证行为不劣化）。

---

## 2. 总体架构变更

### 2.1 关键抽象

引入 **`ScheduledSeq`**：`Scheduler.schedule()` 的返回单元。

```python
@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int         # 本 step 该序列贡献的 token 数（decode=1，prefill_chunk>=1）
    is_prefill_chunk: bool      # 便于后续统计/日志；由 num_new_tokens>1 或 未完成 prompt 判断得出
```

`schedule()` 签名：

```
schedule() -> tuple[list[ScheduledSeq], StepMeta]
StepMeta   -> is_pure_decode: bool, total_new_tokens: int
```

去掉旧的 `(seqs, is_prefill)` 二元返回。

### 2.2 一次 step 的数据流

```
Scheduler.schedule()
  └─ 生成 List[ScheduledSeq]（混合 batch）
     └─ ModelRunner.prepare_step(scheduled)
        ├─ 构造 input_ids / positions / slot_mapping
        ├─ 构造 cu_seqlens_q / cu_seqlens_k / max_seqlen_q / max_seqlen_k
        ├─ 构造 block_tables
        ├─ 【纯 decode】额外填 context_lens（供 flash_attn_with_kvcache 用）
        └─ set_context(...)
     ↓
     ModelRunner.run_model()
     ├─ 若 is_pure_decode & !enforce_eager → CUDA graph fast path（flash_attn_with_kvcache）
     └─ 否则 → eager varlen path（flash_attn_varlen_func）
     ↓
     Sampler → token_ids
     ↓
     Scheduler.postprocess(scheduled, token_ids)
     ├─ decode 序列：append_token
     └─ prefill chunk 序列：num_computed_tokens += num_new_tokens
        ├─ 若本 chunk 是 prompt 最后一段：立即在下一步进入 decode 分支
        └─ 否则保持 running，下一步继续切
```

---

## 3. 数据结构改动

### 3.1 `Sequence`（`engine/sequence.py`）

新增字段：

| 字段 | 语义 | 初始值 |
|---|---|---|
| `num_computed_tokens: int` | 已完成 prefill 的 token 数（进入 attention 参与过计算的 prompt 位置数） | `num_cached_tokens`（前缀 cache 命中的 token 视为已计算） |

派生属性：

```python
@property
def is_prefill_done(self) -> bool:
    return self.num_computed_tokens >= self.num_prompt_tokens

@property
def num_uncomputed_prompt_tokens(self) -> int:
    return max(0, self.num_prompt_tokens - self.num_computed_tokens)
```

**语义澄清**（本次不动 `num_tokens` 的含义）：
- `num_tokens`：物理有效 cache 长度（原语义保留，压缩会用到）。
- `num_computed_tokens`：**逻辑 prefill 推进度**。
- 关系不变：decode 步骤 append_token 时 `num_computed_tokens` 自动跟随 `num_tokens` 增长（因为一个新 decode token 天然也是"已计算"的）。

`append_token` 增加一行：

```python
def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
    self.num_computed_tokens += 1  # 新增
    self.generated_completion_tokens += 1
    self.rope_pos += 1
    self.tail_uncompressed_len += 1
```

`__getstate__` / `__setstate__` 增加 `num_computed_tokens` 的序列化。

### 3.2 `Context`（`utils/context.py`）

- 保留全部现有字段（含压缩相关字段，仍可写入 None）。
- 无新增字段：`cu_seqlens_q/k`、`max_seqlen_q/k` 已存在。
- 增加语义约束（注释）：**decode 分支时 `context_lens` 有效；否则用 `cu_seqlens_k`**。

---

## 4. `Scheduler` 详细设计

### 4.1 数据结构
- `waiting: deque[Sequence]`：新到未开始 prefill 的序列。
- `running: deque[Sequence]`：已至少完成一个 chunk 或已进入 decode 的序列。

**取消 running 内部对 prefill/decode 的显式区分**，靠 `seq.is_prefill_done` 判断。

### 4.2 配置项（新增到 `config.py`）

| 名称 | 默认值 | 语义 |
|---|---|---|
| `long_prefill_chunk_size` | 512 | 单条序列每 step 最多推进的 prefill token 数 |
| `enable_chunked_prefill` | True | 关闭后回退老行为（不切片、prefill 整段） |
| `prefill_first_when_running_empty` | True | `running` 为空时是否用更大预算做 prefill（对首个请求友好） |

### 4.3 调度伪代码

```python
def schedule(self):
    scheduled: list[ScheduledSeq] = []
    token_budget = self.max_num_batched_tokens
    seq_budget   = self.max_num_seqs

    # ---------- (1) decode 优先 ----------
    new_running = deque()
    while self.running and seq_budget > 0 and token_budget > 0:
        seq = self.running.popleft()
        if not seq.is_prefill_done:
            # 还有 prompt 待 prefill 的序列，先放着，(2) 再处理
            new_running.append(seq)
            continue
        # 尝试 append 1 token
        while not self.block_manager.can_append(seq):
            # 内存不足 → 抢占最新加入 running 的一条（LIFO）
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                seq = None
                break
        if seq is None:
            continue
        self.block_manager.may_append(seq)
        scheduled.append(ScheduledSeq(seq, 1, is_prefill_chunk=False))
        new_running.append(seq)
        token_budget -= 1
        seq_budget   -= 1

    # 把剩余（未完成 prefill 的）running 序列还回队列，稍后跟 waiting 一起处理 prefill
    self.running = new_running

    # ---------- (2) prefill chunk 填空 ----------
    # 先处理 running 里未 prefill 完的（正在切片中的老序列），再从 waiting 拉新序列
    pending_prefill = [s for s in self.running if not s.is_prefill_done]
    while (pending_prefill or self.waiting) and seq_budget > 0 and token_budget > 0:
        if pending_prefill:
            seq = pending_prefill.pop(0)
            newly_admitted = False
        else:
            seq = self.waiting[0]
            newly_admitted = True
            if not self.block_manager.can_allocate(seq):
                break  # 装不下就停

        remaining = seq.num_uncomputed_prompt_tokens
        chunk = min(remaining, token_budget, self.long_prefill_chunk_size)
        if chunk <= 0:
            break

        if newly_admitted:
            self.block_manager.allocate(seq)   # 一次分配整条 prompt 的 block
            self.waiting.popleft()
            self.running.append(seq)
            seq.status = SequenceStatus.RUNNING

        scheduled.append(ScheduledSeq(seq, chunk, is_prefill_chunk=True))
        token_budget -= chunk
        seq_budget   -= 1

    assert scheduled or (not self.running and not self.waiting)

    meta = StepMeta(
        is_pure_decode = all(not s.is_prefill_chunk for s in scheduled),
        total_new_tokens = sum(s.num_new_tokens for s in scheduled),
    )
    return scheduled, meta
```

### 4.4 关键设计点

1. **decode 阶段的 `can_append`**：现有语义"再加 1 token 时如果会跨新 block 就检查有没有 free block"，直接沿用。
2. **prefill 阶段的 block 分配策略**：
   - 首次准入时一次性分配整条 prompt 的 block（用现有 `block_manager.allocate(seq)`），**不做 chunk 级分配**。
   - 好处：避免每个 chunk step 都要判分配，`slot_mapping` 计算简单，与前缀 cache/哈希机制一致。
   - 代价：如果整条 prompt 装不下，就直接不准入这条新序列（老逻辑就是这么做的）。
3. **抢占策略保持 LIFO**（沿用现状），仅在 decode 分支触发。
4. **prefill 完成时机**：`postprocess` 里检测 `seq.num_computed_tokens == seq.num_prompt_tokens`，下一 step 自动被 (1) 分支纳入 decode。

### 4.5 `postprocess` 改造

```python
def postprocess(self, scheduled, token_ids):
    # token_ids 只对完成了整段 prompt 的序列有效（即该 step 输出了新 token 的序列）
    # 但 sampler 是对每条序列都会产 1 个 token；对 prefill_chunk 序列该 token 应丢弃
    for sched, tok in zip(scheduled, token_ids):
        seq = sched.seq
        if sched.is_prefill_chunk:
            seq.num_computed_tokens += sched.num_new_tokens
            # prefill chunk 不 append_token, 也不做 EOS/max 判断
        else:
            seq.append_token(tok)
            if (not seq.ignore_eos and tok == self.eos) or \
               seq.generated_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

**关键决策：sampler 仍对整个 batch 出 tokens，但 prefill_chunk 序列的 tok 直接丢弃**。理由：改 sampler 对 batch 做 mask 更复杂，直接丢弃开销可忽略（一次多余的 softmax + argmax）。

---

## 5. `ModelRunner` 改造

### 5.1 合并 `prepare_prefill` / `prepare_decode` → `prepare_step`

```python
def prepare_step(self, scheduled: list[ScheduledSeq]):
    input_ids, positions, slot_mapping = [], [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    max_seqlen_q = max_seqlen_k = 0
    context_lens = []             # 仅纯 decode 时下发
    all_decode = True

    for sched in scheduled:
        seq = sched.seq
        n = sched.num_new_tokens
        if n != 1:
            all_decode = False

        # (a) input_ids: 从 num_computed_tokens 起的 n 个 token
        start = seq.num_computed_tokens
        end   = start + n
        input_ids.extend(seq.token_ids[start:end])

        # (b) positions: rope 位置
        # decode: seq.rope_pos (下一个 token 的位置)
        # prefill chunk: 从 num_computed 位置开始的 n 个位置
        if n == 1:
            positions.append(seq.rope_pos)
        else:
            positions.extend(range(start, end))

        # (c) slot_mapping: n 个物理槽位
        for pos in range(start, end):
            block_idx = pos // self.block_size
            offset    = pos % self.block_size
            slot_mapping.append(seq.block_table[block_idx] * self.block_size + offset)

        # (d) varlen 元数据
        cu_seqlens_q.append(cu_seqlens_q[-1] + n)
        total_k = end  # 该序列本 step 结束后的 KV 长度
        cu_seqlens_k.append(cu_seqlens_k[-1] + total_k)
        max_seqlen_q = max(max_seqlen_q, n)
        max_seqlen_k = max(max_seqlen_k, total_k)

        # (e) 纯 decode 需要 context_lens
        context_lens.append(total_k)

    # transfer to GPU
    input_ids   = to_cuda(input_ids, torch.int64)
    positions   = to_cuda(positions, torch.int64)
    slot_mapping= to_cuda(slot_mapping, torch.int32)
    cu_seqlens_q= to_cuda(cu_seqlens_q, torch.int32)
    cu_seqlens_k= to_cuda(cu_seqlens_k, torch.int32)
    block_tables= self.prepare_block_tables([s.seq for s in scheduled])
    if all_decode:
        context_lens = to_cuda(context_lens, torch.int32)
    else:
        context_lens = None

    set_context(
        is_prefill=False,   # 语义弱化：不再用它做分支，改用 cu_seqlens_q
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,   # 只有纯 decode 时非 None
        block_tables=block_tables,
    )
    return input_ids, positions, all_decode
```

**变化点**：
- 不再有 `prefill` vs `decode` 分支；
- `context.is_prefill` 语义弱化，只在 `Attention.forward` 里辅助判断（见 §6）；
- 上下文 `context_lens` 只在纯 decode 时下发（供 fast-kernel 使用）。

### 5.2 `run_model` 分流

```python
def run_model(self, input_ids, positions, all_decode):
    bs = len(scheduled)  # = 序列数
    ctx = get_context()

    # 只有纯 decode + 未强制 eager + 图能覆盖到 bs 时走 graph
    can_graph = (
        all_decode
        and not self.enforce_eager
        and bs <= self.graph_bs[-1]
    )
    if can_graph:
        return self._run_graph(input_ids, positions, ctx, bs)
    else:
        return self.model.compute_logits(self.model(input_ids, positions))
```

`_run_graph` 就是当前 `run_model` 的图 replay 分支，几乎无改动。

---

## 6. `Attention.forward` 改造

### 6.1 单一 varlen 路径 + decode fast path 分流

```python
def forward(self, q, k, v, Layer):
    ctx = get_context()
    if self.k_cache.numel() and self.v_cache.numel():
        store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)

    # KV 压缩：本期强制不进入
    # if self.kv_compress_enabled and ctx.is_compress_step and ...: MyCompressCompact(...)

    # ---- decode fast path（供 CUDA graph 用）----
    # 判据：context_lens 有值（prepare_step 里只在 all_decode 时才填）
    if ctx.context_lens is not None:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), self.k_cache, self.v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            softmax_scale=self.scale, causal=True,
        )
        return o

    # ---- 混合 batch 或纯 prefill：varlen 统一路径 ----
    o = flash_attn_varlen_func(
        q, self.k_cache, self.v_cache,
        cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
        max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
        softmax_scale=self.scale, causal=True,
        block_table=ctx.block_tables,
    )
    return o
```

### 6.2 关键：`flash_attn_varlen_func` 对 paged cache 的支持
- 当前 prefill 分支已经用它 + `block_table=ctx.block_tables` 走前缀 cache 命中路径，说明该 kernel 版本已支持 paged K/V。
- 混合 batch 情形与之等价：每序列 q_len 可以是 1 或 chunk。**需要在实现前先跑一个 minimal 单元测试确认 varlen kernel 在 q_len=1 时正确性**（预期正确，但要验证）。

### 6.3 CUDA graph 与 varlen 的关系
- Graph 只捕获 `flash_attn_with_kvcache` 路径（现状已是）。
- Varlen 路径始终走 eager，不需要为 varlen 捕图（形状太多，且 prefill chunk 本身 compute-bound，收益低）。
- 兼容性：图捕获时的 `set_context` 参数不含 `cu_seqlens_*`，与新调度不冲突（因为图只会在 `context_lens is not None` 时被选中）。

---

## 7. `BlockManager` 改造

### 7.1 现状问题
`may_append` 假设"每步 1 个新 token"（`len(seq) % block_size == 1` 才分配新 block）。chunked prefill 不适用。

### 7.2 现方案：**prefill 阶段跳过 `may_append`**
因为 §4.4-2 决定 prefill 首次准入时就一次性分配全部 block，chunk step 里 KV 只是写入已分配的槽位，**不需要新增 block**。所以：

- decode 步骤：调 `may_append(seq)`（现状不变）；
- prefill chunk 步骤：**不调** `may_append`。

**结论：`BlockManager` 本次无需改动。**

### 7.3 `can_append` 语义微调
仅 decode 分支调用，无需改。

### 7.4 前缀 cache 影响
- `num_cached_tokens > 0` 的序列首次 chunk 从 `num_cached_tokens` 位置起 prefill（`num_computed_tokens` 初始化为它）。
- 已缓存的 block 不重新写 `slot_mapping`（`prepare_step` 里循环从 `num_computed_tokens` 开始，天然跳过缓存部分）。

---

## 8. 与 KV 压缩的边界

- 本期强制 `kv_compress_enabled=False`（在 example 脚本 / bench 里显式传参）。
- `Attention.forward` 中的压缩分支代码保留但被 `ctx.is_compress_step=False`（`prepare_step` 中显式置 False）短路。
- 后续再启用压缩时需要处理：
  - `compress_selected_batch_indices` 需要基于"decode 子集索引"重新定义（因为 batch 是混合的）；
  - `q_current` 参数改为对应序列的最后一个 token 的 q（在 varlen 语义下需要 gather）；
  - `MyCompressCompact` 依赖的 `context_lens` 需要重新构造。
- 这些工作在本期外，代码里加 TODO 注释。

---

## 9. 分阶段实施计划

| 阶段 | 目标 | 交付物 | 验证 |
|---|---|---|---|
| **P0 基线快照** | 记录当前 baseline 性能 | bench 脚本 + 结果 | 保留一份 log |
| **P1 attention 单路径** | `Attention.forward` 收敛到 varlen + decode fast path 分流；`prepare_prefill/decode` 保持不变但都填 `context_lens=None`/有值以驱动分流 | attention.py, model_runner.py | 精度不变、TTFT/TPOT 无回退 |
| **P2 Sequence + Scheduler 混合调度** | 新增 `num_computed_tokens`；`schedule` 输出 `ScheduledSeq` 混合 batch；`prepare_step` 合并；chunk_size 保守（默认 512） | 全链路走通 | 精度对齐 baseline（同 prompt 同 seed） |
| **P3 CUDA graph 分流** | `run_model` 加纯 decode 分流；确认稳态吞吐≥老 baseline | model_runner.py | 稳态 batch 吞吐持平/优于 baseline |
| **P4 参数扫描 & 调优** | 扫 `long_prefill_chunk_size` ∈ {128, 256, 512, 1024, 2048}；分析 TTFT/TPOT trade-off | 报告 | 在目标场景达到 Pareto 前沿 |
| **P5 压力测试** | 混合请求 workload（长 prompt + 短 prompt 交错到达）、抢占触发、prefix cache 命中 | 稳定性报告 | 无 hang / OOM / 数值异常 |

预计工时：P1~P3 全职 3~5 天，P4~P5 再 2~3 天。

---

## 10. 验证/回归方案

### 10.1 精度验证
- 用相同 seed、相同 prompt、`temperature=0.6`（框架不允许 0）、`max_tokens` 固定：
  - baseline (老 scheduler) vs 新 scheduler (chunk_size=∞ 等效老行为)：**token 序列应严格一致**。
  - baseline vs 新 scheduler (chunk_size=512)：允许因数值路径不同产生细微差异，但 top-1 一致率应 ≥ 99%。
- 数据集：`example.py` 内已有 Math500 prompt，扩展到 20 条。

### 10.2 性能验证
- **稳态吞吐**（waiting 空、running=64 长序列 decode）：应≥ baseline 的 100%（图路径未变）。
- **TTFT under load**（batch=32 到达）：新调度应比 baseline 好 20%+（chunk 化打散长 prompt）。
- **TPOT under load**（batch=32，中间有长 prompt 到达）：新调度应显著更稳定（p99 抖动减少 3~5×）。
- **端到端吞吐**（`bench.py` 场景）：期望 ≥ baseline 95%（chunk 化有轻微开销，可接受）。

### 10.3 单元测试点
- `test_varlen_qlen_1.py`：`flash_attn_varlen_func` 在 q_len=1 时结果与 `flash_attn_with_kvcache` 对齐（数值允许 1e-3 误差）。
- `test_scheduler_mixed.py`：构造 running=2 decode + waiting=1 长 prompt，验证 3 个 step 内的 batch 结构、`num_computed_tokens` 推进正确。
- `test_prefill_chunk_boundary.py`：prompt_len=1000, chunk_size=333，验证 3 个 chunk step + 1 个 decode step 后 token 序列与整段 prefill 一致。

---

## 11. 风险与回滚

| 风险 | 缓解措施 |
|---|---|
| `flash_attn_varlen_func` 在 q_len=1 混合场景有性能陷阱 | P1 里先做微基准；若有 20%+ 回退，`context_lens is not None` 判据可以放宽到 "所有 q_len<=1" 或干脆保留 `flash_attn_with_kvcache` decode 专用路径 |
| chunk 边界处 rope_pos 计算错误 | 单测 `test_prefill_chunk_boundary.py` 覆盖；`positions` 严格用 `range(num_computed_tokens, num_computed_tokens+n)` |
| Sampler 对 prefill_chunk 序列产多余 token 浪费算力 | 一次多余 argmax 开销可忽略；如需严格优化可加 mask，但不在本期 |
| CUDA graph 桶未覆盖 → 混合 batch 频繁触发 eager 拖累稳态 | P3 保证纯 decode 一定走图；混合 batch 走 eager 是设计选择 |
| KV 压缩重启用时接口不兼容 | 本期加 TODO 注释；下一期专门做压缩+混合调度适配 |
| 回滚 | 每一阶段独立 commit；P2 之前的改动可以直接 revert；P2 之后可以通过 `enable_chunked_prefill=False` 走"整段 prefill"退化路径 |

---

## 12. 配置项汇总

`config.py` 新增（其余保持不变）：

```python
# --- chunked prefill / decode-priority scheduling ---
enable_chunked_prefill: bool = True
long_prefill_chunk_size: int = 512
prefill_first_when_running_empty: bool = True   # 保留位，MVP 里可能不用

# --- KV 压缩本期强制关 ---
# 注意：kv_compress_enabled 默认改为 False（在完成 chunked prefill + 压缩联动前）
kv_compress_enabled: bool = False
```

---

## 13. 附：老/新 step() 语义对比

**老逻辑**
```
step():
  seqs, is_prefill = schedule()          # 全 prefill 或全 decode
  token_ids = run(seqs, is_prefill)
  postprocess(seqs, token_ids)
```

**新逻辑**
```
step():
  scheduled, meta = schedule()           # 混合 batch
  token_ids = run(scheduled, meta)       # 内部按 meta.is_pure_decode 分流
  postprocess(scheduled, token_ids)      # decode append_token; prefill 更新 num_computed_tokens
```
