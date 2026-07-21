import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# =============================================================================
# AWQ 权重切分的 factor 说明（重要）
# =============================================================================
# AWQ Linear 持有三个 Parameter：
#   qweight: [in_features, out_features // pack_factor]   int32
#   qzeros:  [in_features // group_size, out_features // pack_factor]  int32
#   scales:  [in_features // group_size, out_features]    float16
#
# 做 TP 切分时，同样的"逻辑 shard_size"（如 num_heads * head_size）需要根据
# 具体切哪个 tensor、沿哪个维度切，除以不同的 factor：
#
#   Tensor    | 沿 dim=0（input 维）切 | 沿 dim=1（output 维）切
#   ----------+------------------------+------------------------
#   qweight   | 不除（dim=0 是原始 in） | 除以 pack_factor
#   qzeros    | 除以 group_size        | 除以 pack_factor
#   scales    | 除以 group_size        | 不除（dim=1 是原始 out）
#
# ColumnParallel 走 dim=1 切分列（对应 output_size）；
# RowParallel   走 dim=0 切分行（对应 input_size）。
# =============================================================================


class AWQLinear(nn.Module):

    def __init__(self, input_size: int, output_size: int, group_size: int, bits: int = 4,
                 bias: bool = False, awq_gemm=None, tp_dim: int | None = None):
        super().__init__()
        self.pack_factor = 32 // bits
        self.group_size = group_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.awq_gemm = awq_gemm

        self.qweight = nn.Parameter(
            torch.empty(input_size, output_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False
        )
        self.qzeros = nn.Parameter(
            torch.empty(input_size // group_size, output_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.empty(input_size // group_size, output_size, dtype=torch.float16),
            requires_grad=False
        )

        self.qweight.weight_loader = self.weight_loader_qweight
        self.qzeros.weight_loader = self.weight_loader_qzeros
        self.scales.weight_loader = self.weight_loader_scales

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.default_weight_loader
        else:
            self.register_parameter("bias", None)

    # 默认 weight_loader（供 bias 等非量化参数使用）
    def default_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_loader_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_loader_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_loader_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x):
        return self.awq_gemm(x, self.qweight, self.qzeros, self.scales,
                             self.group_size, self.pack_factor, bias=self.bias)


class AWQColumnParallelLinear(AWQLinear):
    """
    沿 output 维（qweight 的 dim=1）做 TP 切分。
    output_size 是全量输出维度，内部会除以 tp_size 得到本 rank 持有的大小。
    """

    def __init__(self, input_size: int, output_size: int, group_size: int,
                 bits: int = 4, bias: bool = False, awq_gemm=None):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            group_size, bits, bias,
            awq_gemm=awq_gemm,
            tp_dim=1,   # AWQ ColumnParallel 沿 dim=1（output 维）切
        )

    def weight_loader_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # qweight 沿 dim=1 切，param.size(1) 已经是 out//pack//tp
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = shard_size * self.tp_rank
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_loader_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # qzeros 沿 dim=1 切，同 qweight
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = shard_size * self.tp_rank
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_loader_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # scales 沿 dim=1 切
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = shard_size * self.tp_rank
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)


class AWQMergedColumnParallelLinear(AWQColumnParallelLinear):
    """
    融合层（如 gate_up_proj = gate_proj || up_proj）。
    每个子模块通过 loaded_shard_id 独立加载，共享同一个融合后的 Parameter。
    """

    def __init__(self, input_size: int, output_sizes: list[int], group_size: int,
                 bits: int = 4, bias: bool = False, awq_gemm=None):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), group_size, bits, bias, awq_gemm)

    def _shard_offset_size(self, loaded_shard_id: int, tensor_kind: str):
        """
        计算子模块在融合 param 里的 offset/size（rank-local 视角）。
        tensor_kind: "qweight" | "qzeros" | "scales"

        ColumnParallel 沿 dim=1 切：
          qweight/qzeros 都要除以 pack_factor
          scales 不除
        """
        if tensor_kind in ("qweight", "qzeros"):
            factor = self.pack_factor
        else:  # scales
            factor = 1
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // factor // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // factor // self.tp_size
        return shard_offset, shard_size

    def weight_loader_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "qweight")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_loader_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "qzeros")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_loader_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "scales")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class AWQQKVParallelLinear(AWQColumnParallelLinear):
    """
    q_proj + k_proj + v_proj 融合成 qkv_proj。
    通过 loaded_shard_id ∈ {'q','k','v'} 分别加载。
    """

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, group_size: int,
                 bits: int = 4, total_num_kv_heads: int | None = None, bias: bool = False, awq_gemm=None):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)          # rank-local q head 数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)     # rank-local kv head 数
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, group_size, bits, bias, awq_gemm)

    def _shard_offset_size(self, loaded_shard_id: str, tensor_kind: str):
        """
        计算 q/k/v 子模块在融合 param 里的 offset/size（rank-local 视角）。

        ColumnParallel 沿 dim=1 切：
          qweight/qzeros 除以 pack_factor
          scales 不除

        注意：这里 num_heads/num_kv_heads 已经是 rank-local 的（除过 tp_size），
        所以不需要再除 tp_size。
        """
        if tensor_kind in ("qweight", "qzeros"):
            factor = self.pack_factor
        else:  # scales
            factor = 1

        q_size = self.num_heads * self.head_size // factor
        kv_size = self.num_kv_heads * self.head_size // factor

        if loaded_shard_id == 'q':
            shard_offset = 0
            shard_size = q_size
        elif loaded_shard_id == 'k':
            shard_offset = q_size
            shard_size = kv_size
        elif loaded_shard_id == 'v':
            shard_offset = q_size + kv_size
            shard_size = kv_size
        else:
            raise ValueError(f"loaded_shard_id must be q/k/v, got {loaded_shard_id}")

        return shard_offset, shard_size

    def weight_loader_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "qweight")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_loader_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "qzeros")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_loader_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id, "scales")
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class AWQRowParallelLinear(AWQLinear):
    """
    沿 input 维（qweight 的 dim=0）做 TP 切分，forward 后需要 all_reduce。
    input_size 是全量输入维度，内部会除以 tp_size 得到本 rank 持有的大小。
    """

    def __init__(self, input_size: int, output_size: int, group_size: int,
                 bits: int = 4, bias: bool = False, awq_gemm=None):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            group_size, bits, bias,
            awq_gemm=awq_gemm,
            tp_dim=0,   # AWQ RowParallel 沿 dim=0（input 维）切
        )

    def weight_loader_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # qweight 沿 dim=0 切，param.size(0) 已经是 in//tp
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_loader_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # qzeros 沿 dim=0 切（group 维），param.size(0) 是 (in//tp)//group_size
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_loader_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # scales 沿 dim=0 切（group 维）
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.awq_gemm(x, self.qweight, self.qzeros, self.scales,
                          self.group_size, self.pack_factor, bias=self.bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
