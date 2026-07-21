import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # New property for quantization
    quantization: str | None = None # None or awq
    awq_bits: int = None
    awq_group_size: int = None
    awq_zero_point: bool = None # awq 默认非对称量化
    awq_kernel: str = "torch" # "torch" or "triton"

    #New properties for KV compression
    kv_compress_enabled: bool = True       #Enable KV-cache compression during decode (prefill is not compressed).
    kv_compress_period: int = 1024
    kv_compress_topk: int = 20
    kv_compress_window_blocks: int = 4
    kv_compress_keep_blocks: int = 2
    kv_compress_keep_extra_tokens: int = 1
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.quantization == "awq":
            quant_config_path = os.path.join(self.model, "quantize_config.json")
            if not os.path.isfile(quant_config_path):
                raise FileNotFoundError(f"AWQ quantization config file not found: {quant_config_path}")
            
            import json
            with open(quant_config_path, "r") as f:
                quant_cfg = json.load(f)
                if self.awq_bits is None:
                    self.awq_bits = quant_cfg.get("bits", 4)
                if self.awq_group_size is None:
                    self.awq_group_size = quant_cfg.get("group_size", 128)
                if self.awq_zero_point is None:
                    self.awq_zero_point = quant_cfg.get("zero_point", True)
                
                # 文件内容和显式设置不能冲突
                assert self.awq_bits == quant_cfg.get("bits", self.awq_bits), \
                    f"awq_bits 不匹配: 用户指定 {self.awq_bits}, checkpoint 为 {quant_cfg.get('bits')}"
                assert self.awq_group_size == quant_cfg.get("group_size", self.awq_group_size), \
                    f"awq_group_size 不匹配: 用户指定 {self.awq_group_size}, checkpoint 为 {quant_cfg.get('group_size')}"
                assert self.awq_zero_point == quant_cfg.get("zero_point", self.awq_zero_point), \
                    f"awq_zero_point 不匹配: 用户指定 {self.awq_zero_point}, checkpoint 为 {quant_cfg.get('zero_point')}"
