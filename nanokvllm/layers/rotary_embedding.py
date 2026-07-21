import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_ROPE_CACHE = {}
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    # transformers 5.x 把 rope 配置统一成 dict（如 {'rope_type':'default','rope_theta':...}），
    # 旧实现用 @lru_cache，dict 不可哈希会直接抛 TypeError；这里改成手动缓存并归一化 default 类型。
    if isinstance(rope_scaling, dict):
        if rope_scaling.get("rope_type", "default") != "default":
            raise NotImplementedError(f"unsupported rope_scaling: {rope_scaling}")
        rope_scaling = None
    assert rope_scaling is None
    key = (head_size, rotary_dim, max_position, base)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _ROPE_CACHE[key]
