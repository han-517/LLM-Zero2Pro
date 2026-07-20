"""Week 20：比较缓存与无缓存生成的理论工作量。"""

from __future__ import annotations


def projection_token_count(prompt_tokens: int, new_tokens: int, use_cache: bool) -> int:
    """统计生成 new_tokens 时，被送入 Q/K/V 投影的 token 次数。"""
    # TODO: 无缓存每步重算增长的前缀；缓存只 Prefill 一次并追加新 token。
    raise NotImplementedError


def kv_cache_elements(
    layers: int,
    sequence: int,
    kv_heads: int,
    head_dim: int,
) -> int:
    """返回 K 与 V 合计元素数，忽略 batch。"""
    # TODO: 记得同时计算 K 和 V。
    raise NotImplementedError


if __name__ == "__main__":
    cached = projection_token_count(32, 16, use_cache=True)
    uncached = projection_token_count(32, 16, use_cache=False)
    assert cached == 47
    assert uncached == sum(32 + step for step in range(16))
    assert cached < uncached
    assert kv_cache_elements(12, 128, 4, 64) == 2 * 12 * 128 * 4 * 64
    print("通过：KV Cache 工作量与存储账本")
