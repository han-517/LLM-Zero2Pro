"""在极小内置文本上演示完整训练闭环；不是质量基准。"""

import sys

import torch

from llm_from_scratch.tokenization import BytePairTokenizer
from llm_from_scratch.training import make_next_token_batch, seed_everything
from llm_from_scratch.transformer import GPTConfig, TinyGPT


def main() -> None:
    seed_everything()
    text = ("语言模型通过根据前文预测下一个符号来学习。" * 30) + ("attention reads history. " * 30)
    tokenizer = BytePairTokenizer.train(text, vocab_size=280)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    model = TinyGPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=24,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            d_model=48,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(20260718)
    for step in range(61):
        x, y = make_next_token_batch(tokens, 8, 24, generator=generator)
        _, loss, _ = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step:02d} loss={loss.item():.4f}")

    prefix = torch.tensor([tokenizer.encode("语言")], dtype=torch.long)
    generated = model.generate(prefix, max_new_tokens=30, temperature=0.8, top_k=20)
    decoded = tokenizer.decode(generated[0].tolist(), errors="backslashreplace")
    output_encoding = sys.stdout.encoding or "utf-8"
    printable = decoded.encode(output_encoding, errors="backslashreplace").decode(output_encoding)
    print("生成结果（不完整 UTF-8 字节以转义形式显示）:")
    print(printable)


if __name__ == "__main__":
    main()
