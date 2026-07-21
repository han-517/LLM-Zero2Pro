from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from student_lm import AdamW, ByteBPETokenizer, GPTConfig, TransformerLM, save_checkpoint
from student_lm.training import train_steps

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the project-01 learner-owned language model"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "model.pt")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    corpus_path = PROJECT_ROOT / "data" / "tiny_corpus.txt"
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = ByteBPETokenizer.train([text], vocab_size=280, special_tokens=("<eos>",))
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    torch.manual_seed(args.seed)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    sample_generator = torch.Generator().manual_seed(args.seed + 2)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=176,
        max_seq_len=64,
    )
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = train_steps(
        model,
        token_ids,
        optimizer,
        steps=args.steps,
        batch_size=8,
        sequence_length=32,
        generator=batch_generator,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.output,
        model,
        optimizer,
        step=args.steps,
        generator=batch_generator,
        metadata={
            "config": config.__dict__,
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        },
    )
    tokenizer.save(args.output.with_suffix(".tokenizer.json"))
    prefix = torch.tensor([tokenizer.encode("Language")], dtype=torch.long)
    generated = model.generate(prefix, 40, temperature=0.8, generator=sample_generator)
    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_tokens": args.steps * 8 * 32,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "checkpoint": str(args.output),
        "sample": tokenizer.decode(generated[0].tolist()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
