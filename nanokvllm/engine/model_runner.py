import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer
from nanokvllm.config import Config
from nanokvllm.engine.sequence import Sequence, ScheduledSeq
from nanokvllm.models.qwen3 import Qwen3ForCausalLM
from nanokvllm.layers.sampler import Sampler
from nanokvllm.utils.context import set_context, get_context, reset_context
from nanokvllm.utils.loader import load_model

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config,config)
        load_model(self.model, config.model, quantization=config.quantization)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

        self.kv_compress_enabled = config.kv_compress_enabled
        if config.kv_compress_enabled:#!!!
            self.decode_step_counter = 0

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**24)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

        if hasattr(self, "query_window_manager"):
                    self.query_window_manager.buffers.clear()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """warmup 目的是让 kv_cache 分配的显存估算稳定（跑一次 forward 让 activation 峰值稳定）。
        此时 kv_cache 尚未分配（k_cache/v_cache 是空 tensor），Attention.forward 里
        `if k_cache.numel() and v_cache.numel()` 判定为 False，不会走 store_kvcache，
        因此 slot_mapping / block_tables 传 dummy 即可。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
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

        # varlen 路径 warmup（use_decode_kernel=False）
        set_context(
            False, cu_seqlens_q, cu_seqlens_k,
            max_model_len, max_model_len,
            slot_mapping, None, None,
        )
        self.model(input_ids, positions)
        reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
    
    def prepare_step(self, seqs: list[ScheduledSeq], all_decode: bool):
        """统一 prepare 入口，支持混合 batch（decode + prefill chunk）。

        每条 seq 从 num_computed_tokens 起推进 num_new_tokens 个 token；
        构造 input_ids/positions/slot_mapping 以及 varlen 元数据（cu_seqlens_q/k, max_seqlen_q/k）
        与 decode fast-path 元数据（context_lens）。

        logits_indices: 每条 seq 在拼接 hidden 中的**最后 1 个位置**，供 compute_logits 前 gather 使用
        （修复现有 prefill 采样对齐 bug，同时省掉 lm_head 对 prompt 中间位置的浪费）。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = max_seqlen_k = 0
        slot_mapping = []
        context_lens = []
        logits_indices = []
        running_offset = 0
        for sched in seqs:
            seq = sched.seq
            new_tokens = sched.num_new_tokens
            start = seq.num_computed_tokens
            end = start + new_tokens

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + new_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, new_tokens)
            max_seqlen_k = max(max_seqlen_k, end)

            for pos in range(start, end):
                block_id = seq.block_table[pos // self.block_size]
                slot_mapping.append(block_id * self.block_size + pos % self.block_size)

            context_lens.append(end)
            running_offset += new_tokens
            logits_indices.append(running_offset - 1)

        input_ids_t      = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t      = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t   = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logits_indices_t = torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        block_tables_t   = self.prepare_block_tables([s.seq for s in seqs])

        # 按分流路径决定要转哪些 tensor
        if all_decode:
            # decode fast path（flash_attn_with_kvcache）：只需要 context_lens
            context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_q_t = cu_k_t = None
        else:
            # varlen 路径（flash_attn_varlen_func）：需要 cu_seqlens_*
            cu_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens_t = None

        set_context(all_decode, cu_q_t, cu_k_t, max_seqlen_q, max_seqlen_k,
                    slot_mapping_t, context_lens_t, block_tables_t)
        return input_ids_t, positions_t, logits_indices_t

    # ------------------------------------------------------------------
    # 以下 prepare_prefill / prepare_decode 是老调度器的分离入口，
    # chunked prefill 改造后已被 prepare_step 统一取代，当前未被 run() 调用。
    # 保留原因：prepare_decode 内含 KV 压缩调度逻辑，未来重启用压缩时可参考。
    # 若确认不再需要，可整体删除本注释以下两个方法。
    # ------------------------------------------------------------------
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            
            positions.append(seq.rope_pos)
            # Use `rope_pos` (monotonically increasing, true decode position for RoPE) instead of `len(seq)-1`.
            # KV compression changes the *KV context length* (e.g., S -> R), but RoPE positions must reflect the
            # real token timeline and must NOT shrink/reset after compression.
            #len(seq) now denote the actual compressed seq length,see below
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        # before the upcoming decode step:
        # Determine which sequences have reached the compression trigger (len >= S).
        # Store the decision in the global Context so each Attention layer can: (if needed) compact/compress KV right after store and before flash-attn reads it.

        if self.kv_compress_enabled:
            ctx = get_context()
            # old per-step threshold-trigger compression mask is no longer used as primary trigger
            ctx.compress_need_mask = None
            ctx.compress_any = False

            # -------- periodic compression scheduling --------
            B = self.block_size
            window_blocks = self.config.kv_compress_window_blocks
            window_tokens = window_blocks * B
            topk = self.config.kv_compress_topk

            self.decode_step_counter += 1
            is_compress_step = (self.decode_step_counter % self.config.kv_compress_period == 0)

            selected_batch_indices = []
            selected_seq_ids = []

            if is_compress_step:
                candidates = []
                for i, seq in enumerate(seqs):
                    current_context_len = len(seq)   # current cache length
                    full_blocks = current_context_len // B
                    tail_len = current_context_len % B

                    if tail_len == B - 1:
                        continue
                    elif seq.tail_uncompressed_len >= window_tokens and full_blocks >= window_blocks:
                        candidates.append((i, seq.seq_id))

                # print(candidates,self.decode_step_counter,full_blocks,current_context_len,"model runner")
                selected = candidates[:topk]
                selected_batch_indices = [x[0] for x in selected]
                selected_seq_ids = [x[1] for x in selected]

            ctx.is_compress_step = is_compress_step
            ctx.compress_selected_batch_indices = selected_batch_indices
            ctx.compress_selected_seq_ids = selected_seq_ids
            if is_compress_step and selected_batch_indices:
                ctx.compress_base_context_lens = context_lens.clone()
            else:
                ctx.compress_base_context_lens = None
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, all_decode: bool):
        """统一 forward 入口，**只返回 hidden_states**（不做 compute_logits）。

        compute_logits 由调用方 run() 在 gather 每序列最后位置之后统一做一次，
        避免双重 lm_head 计算。

        分流规则：
        - all_decode=True 且非 eager、bs 在图桶范围、非压缩步骤 → graph fast path
        - 否则 → eager forward（支持 chunked prefill 混合 batch 的 varlen 路径）
        """
        ctx = get_context()
        bs = input_ids.size(0)
        # 压缩步骤强制走 eager（压缩会修改 kv_cache 布局，图 replay 无法处理）
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
        # 图输出是 hidden（capture 时 self.model(...) 只走到 hidden_states，不含 compute_logits）
        return gv["outputs"][:bs]

    @torch.inference_mode()
    def run(self, seqs: list[ScheduledSeq], all_decode: bool):
        input_ids, positions, logits_indices = self.prepare_step(seqs, all_decode)
        temperatures = self.prepare_sample([s.seq for s in seqs]) if self.rank == 0 else None

        hidden = self.run_model(input_ids, positions, all_decode)
        # 每序列取最后 1 个位置的 hidden，再做一次 compute_logits（避免逐 token 走 lm_head）
        last_hidden = hidden.index_select(0, logits_indices)
        logits = self.model.compute_logits(last_hidden)

        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # BEFORE reset_context(), collect compression events (only meaningful on rank 0)
        compression_events = None
        if self.rank == 0:
            ctx = get_context()
            compression_events = ctx.compression_events if hasattr(ctx, "compression_events") else None
        reset_context()
        return token_ids, compression_events

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 关键：图 replay 用于 decode fast path，因此捕图时 use_decode_kernel=True，
            # 让 Attention.forward 在捕图阶段走 flash_attn_with_kvcache 分支。
            # 若传 False 会捕成 varlen 分支，图 replay 时静默走错路径且性能骤降。
            set_context(True,
                        slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
