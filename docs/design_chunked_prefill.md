# Chunked Prefill + Decode 优先调度 实现方案

> 版本：v6（实施完成）
> 作者：nano-kvllm
> 适用代码基线：nano-kvllm v0.2.0
> 参照：vLLM v1 `vllm/v1/core/sched/scheduler.py`
> 状态：**已按 checklist 全部落地**（详见 §9 实施记录）；待用户在 GPU 环境实测验收。

---

## 0. 版本演进与本版定位

- **v4**：从零讲清楚要做什么（设计版）
- **v5**：结合已实现代码，梳理剩余问题 + 修复清单（修订版）
- **v6**（本版）：**实施完成归档**。checklist 已全部勾选；追加 §9 记录实施过程中发现并修复的额外问题（3 处 v5 未识别到的隐藏 bug）、`num_computed_tokens` 语义的最终澄清、最终代码结构与走查结果。

**v5 checklist 状态一览**：见 §4，全部勾选 `[x]`。

**实施中修正的语义（重要）**：
- v5 原方案让 `append_token` 内部 `num_computed_tokens += 1`——**在实施走查时发现语义会冲突**（append 后新 token 尚未过 attention，但字段却说"已计算"，导致下一 step 的 `start = num_computed_tokens` 计算错位）。
- **v6 修正**：`append_token` 只加 `num_tokens`，`num_computed_tokens` 完全由 `Scheduler.postprocess` 显式推进。附录 A 演进表已同步更新。

---

## 1. 修复清单（P0，阻塞跑通）

### 1.1 `scheduler.py`

**bug 1：`is_prefill_done` 是 `@property`，不能当函数调**
```python
# 现状（错）
if not seq.is_prefill_done():
    continue
# 修复
if not seq.is_prefill_done:
    continue
```
影响行：`scheduler.py:36`、`scheduler.py:49`。

**bug 2：`may_append` 传错参数类型**
```python
# 现状（错）
self.block_manager.may_append(ScheduledSeq(seq, 1))
# 修复
self.block_manager.may_append(seq)
```
影响行：`scheduler.py:40`。

**bug 3：`scheduled` 追加的是原始 seq，不是 `ScheduledSeq`**
```python
# 现状（错）
scheduled.append(seq)
# 修复
scheduled.append(ScheduledSeq(seq, 1))
```
影响行：`scheduler.py:41`。

**bug 4：`all_decode` 判据反了**
```python
# 现状（错）：num_new_tokens > 1 才算 decode？逻辑反了
all_decode = all(s.num_new_tokens > 1 for s in scheduled)
# 修复：decode 是 num_new_tokens == 1 且 prefill 已完成
all_decode = all(s.num_new_tokens == 1 and s.seq.is_prefill_done for s in scheduled)
```
影响行：`scheduler.py:84`。

**为什么要检查 `is_prefill_done`**：极端边界——prompt 最后一 chunk 恰好只剩 1 个 token 要 prefill，`num_new_tokens == 1` 但不是 decode，此时不能走 fast path。

**bug 5：`postprocess` 重复 `append_token`**
```python
# 现状（错）：两个分支都 append 了一次，最后又无条件 append 第二次
for sched, token_id in zip(seqs, token_ids):
    seq = sched.seq
    if seq.is_prefill_done:
        seq.append_token(token_id)
    else:
        seq.num_computed_tokens += sched.num_new_tokens
        if seq.is_prefill_done:
            seq.append_token(token_id)
    seq.append_token(token_id)   # ← 多余，删掉
    ...

# 修复
for sched, token_id in zip(seqs, token_ids):
    seq = sched.seq
    if seq.is_prefill_done:
        # decode 步骤：append_token 内部已同步 num_computed_tokens
        seq.append_token(token_id)
    else:
        # prefill chunk：先推进已计算的 prompt token 数
        seq.num_computed_tokens += sched.num_new_tokens
        if seq.is_prefill_done:
            # 本 chunk 刚好补完 prompt：采样出第一个 completion token
            seq.append_token(token_id)
        # 否则是 prefill 中间 chunk，丢弃采样 token（不 append）

    if (not seq.ignore_eos and token_id == self.eos) or \
       seq.generated_completion_tokens >= seq.max_tokens:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
```
影响行：`scheduler.py:121-138`。

### 1.2 `model_runner.py`

**bug 6：`pin_memory` 拼写错**
```python
# 现状（错）
logits_indices = torch.tensor(logits_indices, dtype=torch.int64, pin_memmory=True).cuda(non_blocking=True)
# 修复
logits_indices = torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
```
影响行：`model_runner.py:166`。

**bug 7：`set_context` 前 `cu_seqlens_*` / `context_lens` 未转 CUDA tensor**
```python
# 现状（错）：cu_seqlens_q/k 和 context_lens 还是 python list
set_context(all_decode, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            slot_mapping, context_lens, block_tables)
# 修复：按 all_decode 分流决定要不要转 tensor
if all_decode:
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32,
                                   pin_memory=True).cuda(non_blocking=True)
    cu_q_t = cu_k_t = None
else:
    cu_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32,
                           pin_memory=True).cuda(non_blocking=True)
    cu_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32,
                           pin_memory=True).cuda(non_blocking=True)
    context_lens_t = None
set_context(all_decode, cu_q_t, cu_k_t, max_seqlen_q, max_seqlen_k,
            slot_mapping, context_lens_t, block_tables)
```
影响行：`model_runner.py:169`。

### 1.3 `attention.py`

**bug 8：引用已删除的 `context.is_prefill`**
```python
# 现状（错）：Context 已没有 is_prefill 字段
if (not context.is_prefill)
   and self.kv_compress_enabled
   and context.is_compress_step
   and context.compress_selected_batch_indices:
    MyCompressCompact(...)
# 修复：KV 压缩只在纯 decode 阶段触发（原本设计如此）
if context.use_decode_kernel \
   and self.kv_compress_enabled \
   and context.is_compress_step \
   and context.compress_selected_batch_indices:
    MyCompressCompact(...)
```
影响行：`attention.py:77`。

### 1.4 `llm_engine.py`

**bug 9：`step()` 未适配新签名**
```python
# 现状（错）
def step(self):
    seqs, is_prefill = self.scheduler.schedule()          # ← is_prefill 已废
    ret = self.model_runner.call("run", seqs, is_prefill)
    ...
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)  # ← seqs 是 ScheduledSeq，len(seq) 会崩
    return outputs, num_tokens

# 修复
def step(self):
    scheduled, all_decode = self.scheduler.schedule()
    ret = self.model_runner.call("run", scheduled, all_decode)
    if isinstance(ret, tuple):
        token_ids, compression_events = ret
    else:
        token_ids, compression_events = ret, None
    self.scheduler.postprocess(scheduled, token_ids, compression_events)

    outputs = [(s.seq.seq_id, s.seq.completion_token_ids)
               for s in scheduled if s.seq.is_finished]

    # 吞吐统计：prefill token 数 用正数（每 chunk 贡献 num_new_tokens），decode 用负数（每序列 -1 表征）
    if all_decode:
        num_tokens = -len(scheduled)
    else:
        num_tokens = sum(s.num_new_tokens for s in scheduled)
    return outputs, num_tokens
```
影响行：`llm_engine.py:48-58`。

---

## 2. CUDA graph 路径重构（P0，功能缺失）

### 2.1 现状问题

**问题 A：`run()` 引用了不存在的 `_forward` / `_replay_graph`**

当前 `run()` 里第 315 行：
```python
hidden = self._forward(input_ids, positions, all_decode)
```
而 `_forward` 内部（第 329-333 行）又调 `self._replay_graph(...)`——**`_replay_graph` 方法根本不存在**。这个绕道设计是设计文档里臆造的方法名，实际代码要用的是**现有的 `run_model` 方法**。

**问题 B：`run_model` 内部做了 `compute_logits`，`run()` 又做了一次，导致双重计算**

`run_model` 第 293、307 行：
```python
return self.model.compute_logits(self.model(input_ids, positions))     # 老 prefill 分支
return self.model.compute_logits(graph_vars["outputs"][:bs])            # 老 graph 分支
```
`run()` 第 316-317 行拿到 hidden 后又：
```python
last_hidden = hidden.index_select(0, logits_indices)
logits = self.model.compute_logits(last_hidden)      # ← 第二次 compute_logits
```
**如果 run() 调 run_model 拿到的是 logits（[N, vocab]）**，再 `index_select` 后进 `compute_logits` 就是错的（lm_head 会作用两次）。

**问题 C：`run_model` 的入参 `is_prefill: bool` 与新调度不匹配**

`run_model` 分流判据是 `if is_prefill or ...`，但 `run()` 里已经用 `all_decode`——两者语义完全相反，不能直接传。

**问题 D：`capture_cudagraph` 里 dummy Context 参数错误**

第 353 行：
```python
set_context(False, slot_mapping=..., context_lens=..., block_tables=...)
```
第一个参数 `False` 表示 `use_decode_kernel=False`——但 graph 捕获的正是 decode fast path，应该是 `True`。否则 `Attention.forward` 在捕图阶段走 varlen 分支，捕出来的图是废的（replay 时永远走不到 `flash_attn_with_kvcache` 分支）。

### 2.2 修订方案：改造 `run_model`，删除 `_forward`

**原则**：
- **删除 `_forward` 这个臆造的中间层**
- **改造现有 `run_model`**：接受 `all_decode` 参数、只返回 hidden（不做 compute_logits）
- **`run()` 直接调 `run_model`**

改造后 `run_model`：

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, all_decode: bool):
    """
    统一 forward 入口，返回 hidden_states（不做 compute_logits）。
    compute_logits 由调用方 (run()) 在 gather 每序列最后位置之后统一做。

    分流规则：
    - all_decode=True 且非 eager、bs 在图桶范围、非压缩步骤 → graph fast path
    - 否则 → eager forward（varlen 路径，支持 chunked prefill 混合 batch）
    """
    ctx = get_context()
    bs = input_ids.size(0)
    # 压缩步骤强制走 eager（压缩会修改 kv cache，图 replay 无法处理）
    compress_active = getattr(ctx, "compress_any", False)
    can_graph = (
        all_decode
        and not self.enforce_eager
        and bs <= self.graph_bs[-1]
        and not compress_active
    )

    if not can_graph:
        # eager path：既处理纯 prefill，也处理混合 batch（varlen）
        return self.model(input_ids, positions)

    # ---- graph replay path ----
    graph_bs = next(x for x in self.graph_bs if x >= bs)
    graph = self.graphs[graph_bs]
    gv = self.graph_vars

    gv["input_ids"][:bs]    = input_ids
    gv["positions"][:bs]    = positions
    gv["slot_mapping"].fill_(-1)
    gv["slot_mapping"][:bs] = ctx.slot_mapping
    gv["context_lens"].zero_()
    gv["context_lens"][:bs] = ctx.context_lens
    gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables

    graph.replay()
    return gv["outputs"][:bs]     # 只返回 hidden，交给 run() 做 index_select + compute_logits
```

改造后 `run()`（**删掉 `_forward` 中间层**）：

```python
@torch.inference_mode()
def run(self, seqs: list[ScheduledSeq], all_decode: bool):
    input_ids, positions, logits_indices = self.prepare_step(seqs, all_decode)
    temperatures = self.prepare_sample([s.seq for s in seqs]) if self.rank == 0 else None

    hidden = self.run_model(input_ids, positions, all_decode)          # ← 直接调 run_model
    last_hidden = hidden.index_select(0, logits_indices)               # 每序列取最后 1 位置
    logits = self.model.compute_logits(last_hidden)                    # 一次 compute_logits

    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    compression_events = None
    if self.rank == 0:
        ctx = get_context()
        compression_events = ctx.compression_events if hasattr(ctx, "compression_events") else None
    reset_context()
    return token_ids, compression_events
```

**删除**：`_forward` 方法（臆造的中间层，无用）。

### 2.3 修 `capture_cudagraph`

**关键**：捕图时 dummy Context 的第一个参数应为 `True`（`use_decode_kernel=True`），让 `Attention.forward` 在捕图阶段走 `flash_attn_with_kvcache` 分支。**这是必修的隐蔽 bug——不修则整套图 replay 静默失效**（走了 varlen kernel，性能骤降但不报错）。

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids    = torch.zeros(max_bs, dtype=torch.int64)
    positions    = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs      = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        # ↓↓↓ 关键修复：True，让 Attention 在捕图时走 decode 分支
        set_context(
            True,                                      # use_decode_kernel
            slot_mapping=slot_mapping[:bs],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
        )
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])          # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])      # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    self.graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, outputs=outputs,
    )
```

**注意**：`outputs` 现在存的是 hidden（不是 logits），因为 `model(...)` 只走到 hidden_states，`compute_logits` 由 `run()` 外置。图形状（`[max_bs, hidden_size]`）不变。

### 2.4 warmup 也要适配

**问题**：`warmup_model()` 里调 `self.run(seqs, True)`——但 `run` 的新签名是 `run(scheduled: list[ScheduledSeq], all_decode: bool)`。传 `list[Sequence]` 会崩（`s.seq` 访问失败）；传 `True` 语义反了（warmup 跑的是全长 prefill，不是 decode）。

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    # 用 ScheduledSeq 包装，num_new_tokens=max_model_len 模拟全长 prefill
    # all_decode=False 走 eager varlen 路径（warmup 不需要 graph）
    scheduled = [ScheduledSeq(seq, max_model_len) for seq in seqs]
    self.run(scheduled, all_decode=False)
    torch.cuda.empty_cache()
```

**注意**：warmup 目的是让 kv_cache 分配估算稳定，用假的 `ScheduledSeq(seq, len(prompt))` 模拟全长 prefill 即可，`all_decode=False` 走 eager varlen 路径。

---

## 3. 完整数据流（v5 定稿）

```
LLMEngine.step()
  ├─ scheduled, all_decode = scheduler.schedule()
  │     ├─ (1) 遍历 running：给 prefill 完成的 seq 各分配 1 个 token 做 decode
  │     ├─ (2) 遍历 running：给未完成的 seq 续 chunk（不超 long_prefill_token_threshold）
  │     └─ (3) 拉 waiting 序列：can_allocate 通过 → 一次性分配整条 prompt block，
  │           初始化 num_computed_tokens = num_cached_tokens，做首个 chunk
  │
  ├─ token_ids, compression_events = model_runner.run(scheduled, all_decode)
  │     ├─ prepare_step(scheduled, all_decode)
  │     │     ├─ 从每条 seq 的 num_computed_tokens 起构造 input_ids/positions/slot_mapping
  │     │     ├─ 累积 cu_seqlens_q/k、max_seqlen_q/k、context_lens
  │     │     ├─ 构造 logits_indices（每序列 running_offset-1）
  │     │     └─ set_context(use_decode_kernel=all_decode, ...)
  │     ├─ hidden = run_model(input_ids, positions, all_decode)
  │     │     ├─ all_decode 且可走图 → graph replay，返回 gv["outputs"][:bs]
  │     │     └─ 否则 → self.model(input_ids, positions)      # eager varlen
  │     ├─ last_hidden = hidden.index_select(0, logits_indices)
  │     ├─ logits = model.compute_logits(last_hidden)           # 只做一次 compute_logits
  │     └─ token_ids = sampler(logits, temperatures)
  │
  └─ scheduler.postprocess(scheduled, token_ids, compression_events)
        └─ 按 is_prefill_done 分支（详见 §9.3 与附录 A）：
             ├─ 进入前已 prefill_done（decode 步骤）：
             │     ├─ num_computed_tokens += num_new_tokens（=1）
             │     └─ append_token(tok)      # 只加 num_tokens
             │
             └─ 进入前未 prefill_done（prefill chunk 步骤）：
                   ├─ num_computed_tokens += num_new_tokens
                   └─ 再检查 is_prefill_done：
                        ├─ True（本 chunk 补完 prompt）→ append_token(tok)
                        └─ False（中间 chunk）→ 丢弃采样 token（也不判定 EOS）
```

---

## 4. 全套修复 Checklist

按优先级实施，每一项独立可测。

### P0（必修，跑通所需）— 已全部完成

- [x] `scheduler.py` `is_prefill_done()` → `is_prefill_done`（去括号）
- [x] `scheduler.py` `may_append(ScheduledSeq(...))` → `may_append(seq)`
- [x] `scheduler.py` `scheduled.append(seq)` → `scheduled.append(ScheduledSeq(seq, 1))`
- [x] `scheduler.py` `all_decode` 判据修正为 `num_new_tokens == 1 and s.seq.is_prefill_done`
- [x] `scheduler.py` `postprocess` 删除多余 `append_token`，按 `is_prefill_done` 分支
- [x] `model_runner.py` `pin_memmory` → `pin_memory`
- [x] `model_runner.py` `cu_seqlens_*` / `context_lens` 按分流转 CUDA tensor 后传入 `set_context`
- [x] `attention.py` `not context.is_prefill` → `context.use_decode_kernel`
- [x] `llm_engine.py` `step()` 适配 `(scheduled, all_decode)` 新签名
- [x] 改造 `model_runner.py:run_model`：入参 `all_decode`；只返回 hidden；内联 graph replay
- [x] 删除 `model_runner.py:_forward`（臆造的中间层）
- [x] `run()` 里改 `self._forward(...)` → `self.run_model(...)`
- [x] `capture_cudagraph` `set_context(False, ...)` → `set_context(True, ...)`
- [x] `warmup_model` 重写：绕过 `prepare_step`，直接构造 dummy context 跑 varlen 纯 forward
- [x] `prepare_prefill` / `prepare_decode` 加"已废弃"注释保留（含 KV 压缩参考价值）

### P1（清理）— 已完成

- [x] `Context.is_prefill` 字段已清理
- [x] `embed_head.py` 里老的 last-position gather 逻辑（依赖 `context.is_prefill`）已删除；`get_context` import 一并移除
- [x] `compress_utils.py:127` 里的 `context.is_prefill` 引用改为 `not context.use_decode_kernel`
- [x] `scheduler.schedule()` 已含断言 `scheduled or (not self.running and not self.waiting)`
- [x] `bench.py` / `example.py` 只调 `LLMEngine.generate()` 外层接口，无 `is_prefill` 二元返回引用

### P2（可选，功能完善）— 待补

- [ ] `example_chunked.py` 集成测试脚本：
  - `test_prefill_chunk_boundary`（prompt=1000, chunk=333）
  - `test_scheduler_mixed`（running=2 decode + waiting=1 长 prompt）
  - `test_admission_control`（waiting `can_allocate` 失败不影响 running）

---

## 5. 边界情况澄清

### 5.1 warmup 阶段的 all_decode 语义

warmup 传 `all_decode=False`（走 varlen 路径做全长 prefill），因为：
- warmup 目的是让 kv_cache 分配估算内存 peak，不需要 graph fast path
- 全长 prefill 的 `num_new_tokens == max_model_len > 1`，本来也满足 `all_decode=False`

### 5.2 CUDA graph 捕获期的 Context 不匹配问题

**根源**：graph 捕获时读的是 Context 里 tensor 的**内存地址**，replay 时也读这个地址。所以：
- 捕获时用 dummy tensor（`slot_mapping[:bs]` 等），这些 tensor **必须常驻**（存在 `self.graph_vars` 里）
- Attention.forward 里读 `ctx.slot_mapping` 时，读的其实是 `graph_vars["slot_mapping"][:bs]` 的地址
- Replay 前把真实数据拷进这些常驻 tensor（`gv["slot_mapping"][:bs] = ctx.slot_mapping`）
- 图 replay 时 kernel 从这些常驻 tensor 读输入，输出到 `gv["outputs"]`

**关键点**：`run_model` 走图分支时 **必须先把真实数据拷进 `gv`**，然后**图会自动读 `gv`**（因为捕获时 kernel 就是绑到 `gv` 的地址上的）。这也是为什么图分支返回 `gv["outputs"][:bs]`——它就是图输出的最终归宿。

### 5.3 `use_decode_kernel=True` 时 `cu_seqlens_*` 为 None

`Attention.forward` 的 varlen 分支不会被走到（`ctx.use_decode_kernel` 为 True 直接走 kv_cache 分支），所以 `cu_seqlens_*` 传 None 无害。但**要保证图捕获时也一致**——`set_context` 里 `cu_seqlens_*` 默认 None，图能捕成。

### 5.4 `all_decode` 判据的边界

```python
all_decode = all(s.num_new_tokens == 1 and s.seq.is_prefill_done for s in scheduled)
```

两种情况会让 `all_decode = False`：
- 任何 `num_new_tokens > 1`（prefill chunk）
- 或某个 `num_new_tokens == 1` 但 `is_prefill_done = False`（prompt 最后一 chunk 只剩 1 token）

第二种边界情况罕见但存在，用 `is_prefill_done` 精确排除。

---

## 6. 与原设计（v4）的差异一览

| 项 | v4 | v5 |
|---|---|---|
| Sequence/Scheduler/Attention 主干 | 设计描述 | **已实现，只需修 bug** |
| CUDA graph 集成 | 简述"graph replay 就是现有 run_model" | **改造 `run_model` 接受 `all_decode`、只返回 hidden；删除臆造的 `_forward`；修 `capture_cudagraph` 的 dummy Context** |
| warmup 适配 | 未提 | **明确要用 `ScheduledSeq` 包装** |
| 数据流图 | 抽象数据流 | **对齐当前实际代码路径** |
| 检查清单 | 分阶段实施计划 | **可勾选 Checklist 直接照做** |

v4 已经把设计讲清楚了，v5 就是"根据实际实现进度收敛到可跑通"。

---

## 7. 验证方案

### 7.1 跑通性
1. 单条 prompt（短）：`example.py` 输出通顺
2. 单条长 prompt（>1024 token）：chunked prefill 正常触发，输出与 baseline 一致

### 7.2 混合调度
构造场景：
```python
# 先送一条短 prompt 进 decode 阶段
engine.add_request(short_prompt, sp)
engine.step()   # prefill
for _ in range(5):
    engine.step()   # decode 5 步

# 中间再送一条长 prompt，观察是否混合调度
engine.add_request(long_prompt, sp)
engine.step()   # 应该 all_decode=False（混合），batch 里 1 decode + 1 prefill_chunk
```
验证 `all_decode` 判据、混合 batch 的 varlen kernel 走通、decode 序列不被打断。

### 7.3 精度回归
- baseline vs 新（`long_prefill_token_threshold=99999`）→ 逐 token 一致
- baseline vs 新（`chunk_size=512`）→ top-1 一致率 ≥ 99%

### 7.4 CUDA graph
- 纯 decode batch：`bench.py` 稳态吞吐 ≥ baseline 100%（graph 路径应生效）
- 若性能显著低于 baseline，说明 graph replay 没走通，需要排查 `run_model` 里 `can_graph` 判据 & `capture_cudagraph` 里的 dummy Context 是否传对了 `use_decode_kernel=True`

---

## 8. 配置项汇总

```python
# config.py
enable_chunked_prefill: bool = True         # False 时退化为整段 prefill
long_prefill_token_threshold: int = 512     # 单序列每 step 最多 prefill token 数
kv_compress_enabled: bool = False           # 本期强制关闭
```

---

## 附 A：`num_computed_tokens` 演进示例（v6 修正后）

场景：prompt=1000, prefix 命中 512, chunk=256, decode 3 步

| step | 类型 | num_new_tokens | 进入前 `(num_computed, num_tokens)` | 处理后 `(num_computed, num_tokens)` | 说明 |
|---|---|---|---|---|---|
| 0 | allocate | — | `(0, 1000)` | `(512, 1000)` | `seq.num_computed_tokens = seq.num_cached_tokens` |
| 1 | prefill chunk | 256 | `(512, 1000)` | `(768, 1000)` | postprocess: `num_computed += 256` |
| 2 | prefill chunk | 232 | `(768, 1000)` | `(1000, 1001)` | `num_computed += 232` 补完 prompt，然后 `append_token(tok)` → `num_tokens=1001`（**num_computed 不动**） |
| 3 | decode | 1 | `(1000, 1001)` | `(1001, 1002)` | postprocess: `num_computed += 1` → `1001`，然后 `append_token` → `num_tokens=1002` |
| 4 | decode | 1 | `(1001, 1002)` | `(1002, 1003)` | 同上 |
| 5 | decode | 1 | `(1002, 1003)` | `(1003, 1004)` | 同上 |

**核心不变量**（v6 精确形式）：

```
0 ≤ num_cached_tokens ≤ num_computed_tokens ≤ num_tokens

decode 稳态：num_tokens - num_computed_tokens == 1（上一步 append 的 token 待下一步计算 KV）
prefill 中：  num_tokens - num_computed_tokens == 0（num_tokens 保持 prompt 长度，num_computed 逐 chunk 追赶）
prefill 完成 append 首 completion 后：num_tokens - num_computed_tokens == 1（切入 decode 稳态）
```

**⚠️ 与 v5 的语义差异**（实施中修正）：v5 曾让 `append_token` 内部 `num_computed_tokens += 1`，会破坏"append 后新 token 尚未过 attention"的语义，导致下一 step 的 `start = num_computed_tokens` 错位到"下下个 token"位置。v6 修正为**`append_token` 只加 `num_tokens`**，`num_computed_tokens` 完全由 `Scheduler.postprocess` 显式推进。

---

## 附 B：老 / 新 step() 语义对比

**老**
```python
seqs, is_prefill = scheduler.schedule()          # 全 prefill 或全 decode
token_ids = model_runner.run(seqs, is_prefill)   # 采样对齐 bug 潜伏
scheduler.postprocess(seqs, token_ids)
```

**新（v5 收尾后）**
```python
scheduled, all_decode = scheduler.schedule()          # 混合 batch
token_ids, comp_evts = model_runner.run(scheduled, all_decode)
                                                       # 内部走 graph fast path / eager varlen
                                                       # logits_indices gather 修复采样对齐
scheduler.postprocess(scheduled, token_ids, comp_evts) # 按 is_prefill_done 分支推进
```

---

## 附 C：为什么 running 队列混装 prefill 和 decode 序列

一个常见误解是"vLLM 把未 prefill 完的塞回 waiting，我们不应该也这样吗？"—— **不是**。

vLLM v1 的 `self.running` 里同时装 prefill_chunk 中和 decode 中的所有序列（源码 `scheduler.py:465` "First, schedule the RUNNING requests" 一段就是同时处理两种）。`is_prefill_chunk` 是 `Request` 的**动态属性**（`num_computed_tokens < num_tokens`），不是队列成员。

**队列的语义是"准入状态"，不是"prefill/decode 阶段"**：
- `waiting`：从未准入过的新序列（`num_computed_tokens == 0`）
- `running`：已准入的所有序列，用 `is_prefill_done` 属性区分阶段

理由：chunked prefill 需要**混合 batch**（同一 step 里 decode + prefill_chunk 并存）。如果按阶段分队列，跨队列组 batch 逻辑上很混乱；一个队列 + 一个 property 是最简洁的表达。

nano-kvllm v5/v6 沿用这个语义。

---

## 9. 实施记录（v6 落地）

本节记录实施过程中每个改动的最终形态、遇到并修复的额外问题、以及全链路走查结果。定位为"落地档案"，方便未来回顾与新人上手。

### 9.1 最终文件改动一览

| 文件 | 改动性质 | 关键改动点 |
|---|---|---|
| `nanokvllm/engine/sequence.py` | **语义修正** | `append_token` 不再动 `num_computed_tokens`；保留 `is_prefill_done` / `num_uncomputed_prompt_tokens` property |
| `nanokvllm/engine/scheduler.py` | Bug 修复 | `is_prefill_done` 去括号；`may_append`/`scheduled` 类型修正；`all_decode` 判据翻正；`postprocess` 按 `is_prefill_done` 分支 + 显式推进 `num_computed_tokens` + `appended` 标记避免误判 EOS |
| `nanokvllm/engine/model_runner.py` | 大幅重构 | `prepare_step` 修拼写 + 补 CUDA 转换；`run_model` 改造入参 `all_decode` + 只返回 hidden + 内联 graph replay；删除 `_forward`；**重写 `warmup_model`**（绕过 prepare_step，避免 block_table 空导致的 IndexError）；`capture_cudagraph` 修 dummy Context 参数 |
| `nanokvllm/layers/attention.py` | 引用修正 | `not context.is_prefill` → `context.use_decode_kernel`（KV 压缩分支判据） |
| `nanokvllm/utils/context.py` | 已由用户完成 | 新增 `use_decode_kernel` 字段（本节实施前已就绪） |
| `nanokvllm/engine/llm_engine.py` | 接口适配 | `step()` 适配 `(scheduled, all_decode)`；`num_tokens` 统计从 `sum(len(seq))` 改为 `sum(s.num_new_tokens)`；`is_finished` 判定改为 `s.seq.is_finished` |
| `nanokvllm/layers/embed_head.py` | **P1 清理** | 删除 `ParallelLMHead.forward` 里对整段 hidden 做 last-position gather 的老逻辑（已上移到 `run()` 里 `hidden.index_select(0, logits_indices)`）；一并删除 `get_context` import |
| `nanokvllm/layers/compress_utils.py` | 引用修正 | `context.is_prefill` → `not context.use_decode_kernel` |
| `nanokvllm/config.py` | 配置默认值 | `kv_compress_enabled` 默认 `True` → `False`（本期不联动） |

### 9.2 实施中发现并修复的 3 个 v5 未识别问题

**额外 bug 1：`embed_head.py:58-60` 里老的 last-position gather 与新 `logits_indices` 语义冲突**

`ParallelLMHead.forward` 老代码里：
```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
logits = F.linear(x, self.weight)
```

这个 gather 是为老"整段 prefill → 只对最后位置采样"服务的。**v5 里我们已经在 `run()` 里通过 `hidden.index_select(0, logits_indices)` 完成了这个 gather**——若不删除 embed_head 里的 gather，会导致**双重 gather**（先 index_select 取每序列最后位置，再进 compute_logits 走一次 cu_seqlens 索引，越界崩溃）。修复：**完全删除 embed_head 里的 gather 逻辑**，`compute_logits` 接收的 `x` 已是 `[num_seqs, hidden]`，直接线性投影。

**额外 bug 2：`compress_utils.py:127` 里隐藏的 `context.is_prefill` 引用**

grep 时才发现。此处判据：
```python
if context.is_prefill or context.context_lens is None or context.block_tables is None:
    return False
```
用于压缩逻辑的短路。修复：`not context.use_decode_kernel` 等价替换。

**额外 bug 3：`scheduler.postprocess` 的 compression_events 分支里 `seqs[bidx]` 直接当 Sequence 用**

原代码：
```python
for bidx, ev in dedup.items():
    seq = seqs[bidx]                             # ← 现在 seqs 是 list[ScheduledSeq]，seq 是 ScheduledSeq
    self.block_manager.truncate_blocks(seq, ...) # ← 崩
```

本期 `kv_compress_enabled=False` 时不会进入这个分支，但作为 defensive 修复：`seq = seqs[bidx].seq`。

### 9.3 v6 修正的关键设计决策：`num_computed_tokens` 与 `append_token` 的关系

v5 让 `append_token` 内部 `num_computed_tokens += 1`，看似"新 token 也算已计算"。**实施走查追踪一条 prompt=1200 seq 的生命周期时发现严重错位**：

**v5 语义（错误）**下的 Step 4（第一次 decode）：
- Step 3 postprocess 结束：`num_tokens=1201`, `num_computed_tokens=1201`（append 内 +1）
- Step 4 prepare_step: `start = num_computed_tokens = 1201`, `end = 1202`
- `input_ids = seq[1201:1202]`——但 `token_ids` 只有 1201 个元素，切片返回**空列表**，forward 崩溃

**v6 语义（正确）**下的 Step 4：
- Step 3 postprocess 结束：`num_tokens=1201`, `num_computed_tokens=1200`（append 不动 num_computed）
- Step 4 prepare_step: `start = 1200`, `end = 1201`
- `input_ids = seq[1200:1201] = [新 append 的 token]`，正是"要为它计算 KV 并采样下一 token" ✓

**核心洞察**：`num_computed_tokens` 表示"KV cache 中已经写入的 token 数"，不是"逻辑上已产生的 token 数"。append_token 只让 `num_tokens` 前进（token 已产生），KV 计算发生在下一 step 的 prepare_step + forward + postprocess 的推进里。

#### 9.3.1 `num_computed_tokens` 的完整更新路径

v6 里 `num_computed_tokens` 只在 **3 个位置** 被写入，其余位置只读。这个约束换来了字段语义的清晰性。

**写入位置 1：准入时初始化（`scheduler.py` schedule 阶段 (3)）**

```python
# 从 waiting 拉新序列，分配 block 后立刻初始化 num_computed_tokens
self.block_manager.allocate(seq)                  # 内部会设置 seq.num_cached_tokens
seq.num_computed_tokens = seq.num_cached_tokens   # ← 命中的 prefix 视为"已计算"
seq.status = SequenceStatus.RUNNING
```

含义：**prefix cache 命中的 block 里已经存有正确 KV**，等价于这些 token 已经过了 attention 计算，直接把 `num_computed_tokens` 拉到命中位置，省掉这段 prefill。

**写入位置 2：postprocess 里推进 chunk / decode 消费的 token 数（`scheduler.py:postprocess`）**

```python
if seq.is_prefill_done:
    # decode 步骤：本 step 消费了 seq.token_ids[num_computed:num_computed+1]（=上一步 append 的 token）
    seq.num_computed_tokens += sched.num_new_tokens   # =1
    seq.append_token(token_id)                        # 只加 num_tokens
else:
    # prefill chunk：本 step 消费了 seq.token_ids[num_computed:num_computed+n]（= 一段 prompt）
    seq.num_computed_tokens += sched.num_new_tokens
    if seq.is_prefill_done:
        # 本 chunk 刚好补完 prompt：采样出的是第一个 completion token
        seq.append_token(token_id)                    # 只加 num_tokens
    # 否则中间 chunk，丢弃采样
```

含义：**只要本 step 有 `num_new_tokens` 个 token 完成了 attention（KV 写进 cache），就把这个数加到 `num_computed_tokens` 上**。两种情形合并为一条更新：`num_computed_tokens += num_new_tokens`。

**为什么两条分支都用 `+= num_new_tokens`**——因为 `num_new_tokens` 精确等于本 step forward 里"从 `num_computed_tokens` 起写入 KV cache 的 token 数"：
- decode 时 `num_new_tokens=1`（就是上一步 append 的那个 token 现在被计算 KV）
- prefill chunk 时 `num_new_tokens=n`（本 chunk 消费的 prompt 段）

这条更新在 postprocess 早期做完，**之后再调 `append_token(token_id)` 只影响 `num_tokens`**，让 decode 稳态下的不变量 `num_tokens - num_computed_tokens == 1` 得以维持。

**写入位置 3：抢占重算（`scheduler.py:preempt`，当前简化版未主动调用，但保留）**

```python
def preempt(self, seq):
    ...
    seq.num_computed_tokens = 0    # 抢占后重新从头 prefill
    self.block_manager.deallocate(seq)
    ...
```

含义：抢占本质是"丢弃这条 seq 的 KV cache，重新排队跑 prefill"，`num_computed_tokens` 归零表示"没有任何 token 的 KV 存在于 cache 中"。

#### 9.3.2 三个更新位置的语义闭合验证

用附录 A 的场景（prompt=1000, prefix 命中 512, chunk=256）反推：

| 事件 | 写入位置 | num_computed_tokens | 语义验证 |
|---|---|---|---|
| add_request 后 | 未写 | 0 | seq 未准入，无 KV，`= 0` ✓ |
| waiting → running (allocate) | 位置 1 | 512 | prefix cache 命中 2 个 block（`256*2=512`），KV 已在 cache 里 ✓ |
| Step 1 postprocess（chunk 256） | 位置 2 | 768 | 本 step 写入 pos [512..767] 的 KV，累计 768 ✓ |
| Step 2 postprocess（chunk 232 补完 prompt） | 位置 2 | 1000 | 本 step 写入 pos [768..999]，累计 1000；append_token 只加 num_tokens → (1000, 1001) ✓ |
| Step 3 postprocess（decode） | 位置 2 | 1001 | 本 step 写入 pos [1000..1000] 的 KV（上一步 append 的 completion token），累计 1001；append_token → (1001, 1002) ✓ |
| Step 4 decode | 位置 2 | 1002 | 同上，稳态 ✓ |

**每一步的 `num_computed_tokens` 值都精确对应"kv_cache 中已写入的 token 数"**——这个不变量在整个生命周期里保持。

#### 9.3.3 `num_computed_tokens` 的所有读取位置

作为对照，列出所有**读**这个字段的地方，帮助理解字段的下游用途：

| 位置 | 读取用途 |
|---|---|
| `Sequence.is_prefill_done` property | 判断是否已完成 prefill（`num_computed_tokens >= num_prompt_tokens`） |
| `Sequence.num_uncomputed_prompt_tokens` property | 供 `_compute_chunk_size` 计算本 step 还能做多少 prefill |
| `Scheduler._compute_chunk_size` | 计算本 step 的 `num_new_tokens` 上限 |
| `ModelRunner.prepare_step` | `start = seq.num_computed_tokens`，构造 input_ids / positions / slot_mapping / cu_seqlens_k 的起点 |
| `Scheduler.postprocess` | 推进本字段（写入位置 2），并再次读取以判断本 chunk 是否已补完 prompt |

**所有读取都在获取"下一步该从哪个位置开始计算 KV"这个信息**——语义纯粹，字段承担单一职责。

### 9.4 warmup 阶段的处理（对齐现有代码习惯）

**问题**：新 `prepare_step` 里 `for pos in range(start, end): seq.block_table[pos // block_size]` 对未 allocate 的 seq 会 IndexError（block_table=[]）。老 `prepare_prefill` 有"未 allocate 跳过 slot_mapping"的 warmup 分支，新 `prepare_step` 没有这个特判。

**选项对比**：
- A. 给 `prepare_step` 加"block_table 空则跳过"分支（污染主逻辑）
- B. warmup 时给 seq 分配假 block_table（引入无意义 block）
- C. **warmup 完全绕过 `prepare_step`，直接构造 dummy context 跑纯 forward**（选此）

**warmup 的最终实现**：
```python
def warmup_model(self):
    """warmup 目的：让 kv_cache 分配的显存估算稳定。
    此时 kv_cache 尚未分配（k_cache/v_cache.numel()==0），Attention.forward 里
    `if k_cache.numel() and v_cache.numel()` 判定为 False → 不走 store_kvcache，
    因此 slot_mapping / block_tables 传 dummy 即可。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    if num_seqs == 0:
        torch.cuda.empty_cache()
        return

    total_tokens = num_seqs * max_model_len
    input_ids = torch.zeros(total_tokens, dtype=torch.int64, device="cuda")
    positions = torch.cat([
        torch.arange(max_model_len, dtype=torch.int64, device="cuda")
        for _ in range(num_seqs)
    ])
    cu_seqlens_q = torch.tensor(
        [i * max_model_len for i in range(num_seqs + 1)],
        dtype=torch.int32, device="cuda",
    )
    cu_seqlens_k = cu_seqlens_q.clone()
    slot_mapping = torch.full((total_tokens,), -1, dtype=torch.int32, device="cuda")

    # 走 varlen 路径 warmup（use_decode_kernel=False）
    set_context(False, cu_seqlens_q, cu_seqlens_k, max_model_len, max_model_len,
                slot_mapping, None, None)
    self.model(input_ids, positions)
    reset_context()
    torch.cuda.empty_cache()
```

**语义闭合验证**：
- k_cache 为空 tensor → 跳过 `store_kvcache`，slot_mapping 无效值 `-1` 无影响 ✓
- `block_tables=None`，`Attention.forward` 里 `if context.block_tables is not None: k, v = k_cache, v_cache` 不触发，仍用 QKV proj 的 local k/v ✓
- 走 `flash_attn_varlen_func` 纯 attention 计算（无 paged read），仅为让 activation 峰值稳定

### 9.5 CUDA graph 路径最终实现

**`run_model` 完整代码**：

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, all_decode: bool):
    """统一 forward 入口，只返回 hidden_states（不做 compute_logits）。
    compute_logits 由调用方 run() 在 gather 每序列最后位置之后统一做一次。"""
    ctx = get_context()
    bs = input_ids.size(0)
    compress_active = getattr(ctx, "compress_any", False)
    can_graph = (
        all_decode
        and not self.enforce_eager
        and bs <= self.graph_bs[-1]
        and not compress_active
    )
    if not can_graph:
        return self.model(input_ids, positions)

    # ---- graph replay path ----
    graph_bs = next(x for x in self.graph_bs if x >= bs)
    graph = self.graphs[graph_bs]
    gv = self.graph_vars
    gv["input_ids"][:bs] = input_ids
    gv["positions"][:bs] = positions
    gv["slot_mapping"].fill_(-1)
    gv["slot_mapping"][:bs] = ctx.slot_mapping
    gv["context_lens"].zero_()
    gv["context_lens"][:bs] = ctx.context_lens
    gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
    graph.replay()
    return gv["outputs"][:bs]
```

**`capture_cudagraph` 关键片段**：
```python
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    # 关键：True，让 Attention 在捕图时走 decode 分支
    set_context(True,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
    ...
```

**`outputs` 现在存的是 hidden**（`Qwen3ForCausalLM.forward` 只返回 hidden），维度 `[max_bs, hidden_size]` 不变；`compute_logits` 完全外置到 `run()`。

### 9.6 全链路走查（prompt=1200 tokens, chunk=512）

追踪 seq 完整生命周期，验证每 step 状态正确。

| step | 类型 | schedule 走支路 | prepare start~end | forward path | postprocess 结果 `(num_computed, num_tokens)` |
|---|---|---|---|---|---|
| add_request | — | — | — | — | `(0, 1200)`, running.append after (3) |
| 1 | prefill chunk 512 | (3) waiting → running, allocate 5 blocks | 0~512 | eager varlen | `(512, 1200)` |
| 2 | prefill chunk 512 | (2) 续 chunk | 512~1024 | eager varlen | `(1024, 1200)` |
| 3 | prefill chunk 176（最后一 chunk） | (2) 续 chunk | 1024~1200 | eager varlen | `(1200, 1201)`——append 首 completion |
| 4 | decode | (1) 已 prefill_done | 1200~1201 | **graph replay**（all_decode=True） | `(1201, 1202)` |
| 5+ | decode | (1) | `num_computed`~`num_computed+1` | graph replay | 每步 +1，直到 EOS 或 max_tokens |

**关键 slot_mapping 验证**（block_size=256, 5 个 block 分配给 seq）：
- Step 1 pos 0..511 → block 0/1 的槽位 0..511（写 kv_cache）
- Step 2 pos 512..1023 → block 2/3 的槽位 0..511（写 kv_cache）
- Step 3 pos 1024..1199 → block 4 的槽位 0..175（写 kv_cache）
- Step 4 pos 1200 → block 4 的槽位 176（写 kv_cache）——**仍在 block 4，不需新 block**
- Step N（decode）：`may_append` 在 `len(seq) % block_size == 1` 时才追加新 block ✓

**Step 2 的混合前缀读**：`cu_seqlens_q=[0,512]` 但 `cu_seqlens_k=[0,1024]`——Q 只算本 chunk 的 512 token，K/V 读全 1024 token（前 512 从 Step 1 写入的 kv_cache 里读，通过 `block_table` paged 访问）。这是 chunked prefill 的核心场景，`flash_attn_varlen_func + block_table` 支持 ✓。

### 9.7 潜在遗留与已知限制

**保留但不使用的代码**：
- `model_runner.py:prepare_prefill` / `prepare_decode`：加"已废弃"注释保留，`prepare_decode` 内含 KV 压缩调度参照价值
- `scheduler.py:preempt`：本期简化版不做主动抢占，方法保留待未来升级

**已知限制**：
- **KV 压缩本期禁用**（`kv_compress_enabled=False`）；未来重启用需在 `prepare_step` 里补齐 `compress_*` 字段并做混合 batch 适配
- **抢占仅做准入控制**（`can_allocate` 失败则 back-pressure，不 pop running）；极端 OOM 场景下可能有序列停摆，但不 hang
- **CUDA graph 只覆盖纯 decode batch**；混合 batch 走 eager varlen（性能有小幅开销，但是设计决策）

### 9.8 待验收清单（用户在 GPU 环境实测）

按优先级实测：

1. **基础跑通**：`python example.py` 短 prompt 输出通顺
2. **chunked prefill 触发**：跑 `prompt_len > 512` 长 prompt，观察调度 log 是否出现多个 chunk step；输出与整段 prefill 语义一致
3. **精度回归**：`long_prefill_token_threshold=99999`（等效整段）vs `=512`，同 seed 下 top-1 一致率应 ≥ 99%
4. **CUDA graph 生效**：`enforce_eager=False` 下跑纯 decode 稳态，`bench.py` 吞吐应 ≥ baseline 100%（若显著下降说明 graph 未生效，排查 `run_model:can_graph` 判据 & `capture_cudagraph` 的 dummy Context）
5. **混合调度**：短 prompt 进 decode 后再送长 prompt，观察是否混合 batch（`all_decode=False`）且 decode 序列不被打断

**故障排查优先级**（若出问题）：
1. **图 replay 静默失效**：检查 `capture_cudagraph` 里 `set_context(True, ...)` 是否生效（应看到 Attention 走 `flash_attn_with_kvcache` 分支）
2. **slot_mapping 越界**：打印 `context.slot_mapping` 与 `seq.block_table`，确认 `pos // block_size` 索引在 `len(block_table)` 内
3. **精度偏差过大**：单序列同 seed 下比对 `logits[0]` 是否与 baseline 逐位置对齐（关注 `hidden.index_select` 结果的 contiguity）
