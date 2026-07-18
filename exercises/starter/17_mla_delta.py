"""Week 31/33：区分 latent cache accounting 与 Delta 写入规则。"""

from torch import Tensor


def latent_cache_bytes(
    batch: int, sequence: int, latent_dim: int, *, bytes_per_element: int = 2
) -> int:
    """只计算持久 latent cache，不把临时重建的 K/V 伪装成缓存节省。"""
    # TODO: 验证所有维度和 dtype 字节数。
    raise NotImplementedError


def delta_update(state: Tensor, key: Tensor, value: Tensor, beta: Tensor) -> Tensor:
    """实现 S <- S + beta * (v - S k) outer k。"""
    # TODO: 支持 batch，并保持 state 输入不被原地修改。
    raise NotImplementedError
