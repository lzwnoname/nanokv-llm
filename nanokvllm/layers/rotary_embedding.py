from functools import lru_cache

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


def _freeze(obj):
    """递归把 dict / list / set 转成可哈希对象，作为 lru_cache 的 key。

    transformers 5.x 起 rope_scaling 等配置以 dict 形式下发，dict 本身不可哈希，
    直接调 lru_cache 会抛 TypeError。这里做一次通用归一化，未来配置结构变化也不用改 get_rope。
    """
    if isinstance(obj, dict):
        return frozenset((k, _freeze(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_freeze(v) for v in obj)
    return obj


@lru_cache(maxsize=8)
def _get_rope_cached(head_size, rotary_dim, max_position, base, _rope_scaling_key):
    # _rope_scaling_key 只用于参与哈希，本实现仅支持 default 类型（已在外层校验）
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if isinstance(rope_scaling, dict):
        if rope_scaling.get("rope_type", "default") != "default":
            raise NotImplementedError(f"unsupported rope_scaling: {rope_scaling}")
        rope_scaling = None
    assert rope_scaling is None
    return _get_rope_cached(head_size, rotary_dim, max_position, base, _freeze(rope_scaling))
