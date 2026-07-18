import torch

from llm_from_scratch.attention import GroupedQueryAttention, MultiHeadAttention, SimpleMLA


def main() -> None:
    batch, time, width, query_heads, kv_heads = 2, 128, 64, 8, 2
    x = torch.randn(batch, time, width)
    mha = MultiHeadAttention(width, query_heads)
    gqa = GroupedQueryAttention(width, query_heads, kv_heads)
    mla = SimpleMLA(width, query_heads, latent_dim=16)

    _, mha_cache = mha(x)
    _, gqa_cache = gqa(x)
    _, latent_cache = mla(x)
    print("MHA cache elements:", sum(item.numel() for item in mha_cache))
    print("GQA cache elements:", sum(item.numel() for item in gqa_cache))
    print("Teaching MLA latent cache elements:", latent_cache.numel())
    print("注意：元素数只描述缓存，不是完整质量或墙钟基准。")


if __name__ == "__main__":
    main()

