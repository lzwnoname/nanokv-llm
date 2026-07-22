from collections import deque

from nanokvllm.config import Config
from nanokvllm.engine.sequence import Sequence, SequenceStatus, ScheduledSeq
from nanokvllm.engine.block_manager import BlockManager
from transformers import AutoTokenizer

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()#!!!
        self.running: deque[Sequence] = deque()#!!!
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[ScheduledSeq], bool]:
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs
        scheduled: list[ScheduledSeq] = []
        
        # decode 优先
        for seq in self.running:
            if seq_budget <= 0 or token_budget <= 0:
                break
            if not seq.is_prefill_done:
                continue
            if not self.block_manager.can_append(seq):
                # 简化版：装不下就跳过（不抢占），下 step 再试
                continue
            self.block_manager.may_append(seq)
            scheduled.append(ScheduledSeq(seq, 1))
            token_budget -= 1
            seq_budget -= 1

        # 给 running 里未prefill完的续chunk
        for seq in self.running:
            if seq_budget <= 0 or token_budget <= 0:
                break
            if seq.is_prefill_done:
                continue
            new_tokens = self._compute_chunk_size(seq, token_budget)
            if new_tokens <= 0:
                continue
            scheduled.append(ScheduledSeq(seq, new_tokens))
            token_budget -= new_tokens
            seq_budget -= 1
        
        # 调度waiting队列
        while self.waiting and seq_budget > 0:
            seq = self.waiting[0]
            # 和vllm一样准入控制，能放下整个prompt kv cache才放入
            if not self.block_manager.can_allocate(seq):
                break
            # 一次性分配整个 prompt 的block
            self.block_manager.allocate(seq)
            seq.num_computed_tokens = seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            
            new_tokens = self._compute_chunk_size(seq, token_budget)
            if new_tokens <= 0:
                self.block_manager.deallocate(seq)
                seq.num_computed_tokens = 0
                seq.status = SequenceStatus.WAITING
                break
            
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(ScheduledSeq(seq, new_tokens))
            token_budget -= new_tokens
            seq_budget -= 1
        
        assert scheduled or (not self.running and not self.waiting)

        # 边界：prompt 最后一 chunk 只剩 1 token 时 num_new_tokens==1 但 is_prefill_done=False，
        # 这种情况不能走 decode fast path，必须 all_decode=False（走 varlen）
        all_decode = all(
            s.num_new_tokens == 1 and s.seq.is_prefill_done for s in scheduled
        )
        return scheduled, all_decode

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        
        
        # restore logical sequence length before recompute
        seq.num_tokens = len(seq.token_ids)
        if len(seq.token_ids) > 0:
            seq.last_token = seq.token_ids[-1]
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        
    def postprocess(self, seqs: list[ScheduledSeq], token_ids: list[int], compression_events: list | None = None) -> list[bool]:
        if compression_events:
            # Deduplicate by batch_index: only keep the last event for each seq
            dedup = {}
            for ev in compression_events:
                bidx = ev["batch_index"]
                if 0 <= bidx < len(seqs):
                    dedup[bidx] = ev

            for bidx, ev in dedup.items():
                # seqs 现在是 list[ScheduledSeq]，取内部 Sequence
                seq = seqs[bidx].seq

                new_context_len = ev["new_context_len"]
                keep_blocks = ev["keep_blocks"]

                self.block_manager.truncate_blocks(seq, keep_blocks)

                # num_tokens still denotes current cache length in your current design
                seq.num_tokens = new_context_len

                # periodic compression: reset newly-grown-token counter
                seq.tail_uncompressed_len = ev.get("tail_uncompressed_len_after", 0)
        
        for sched, token_id in zip(seqs, token_ids):
            seq = sched.seq
            appended = False

            if seq.is_prefill_done:
                # decode 步骤：本 step 已把 seq.last_token（上一 step append 的那个）的 KV 写入 cache，
                # 现在推进 num_computed_tokens += 1 表示"这个 token 已被计算"，
                # 然后 append_token 加入这次采样出的新 token（新 token 待下一 step 再计算）。
                seq.num_computed_tokens += sched.num_new_tokens  # =1
                seq.append_token(token_id)
                appended = True
            else:
                # prefill chunk：先推进本 step 计算的 prompt token 数
                seq.num_computed_tokens += sched.num_new_tokens
                if seq.is_prefill_done:
                    # 本 chunk 刚好把 prompt 补完：采样出的是第一个 completion token（待下一 step decode）
                    seq.append_token(token_id)
                    appended = True
                # 否则是 prefill 中间 chunk，丢弃采样 token（不 append，也不判定结束）

            # 只有真正 append 了 completion token 才做结束判定
            # （丢弃的 prefill 中间 token 就算恰好等于 eos 也不能结束序列）
            if appended and (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.generated_completion_tokens >= seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def _compute_chunk_size(self, seq: Sequence, token_budget: int):
        remaining_tokens = seq.num_uncomputed_prompt_tokens
        if self.enable_chunked_prefill and self.long_prefill_token_threshold > 0:
            remaining_tokens = min(remaining_tokens, self.long_prefill_token_threshold)
        return min(remaining_tokens, token_budget)