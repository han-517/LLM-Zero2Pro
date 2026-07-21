from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT01_ROOT = PROJECT_ROOT.parent / "01_end_to_end_lm"
if str(PROJECT01_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT01_ROOT))

from student_alignment import (  # noqa: E402
    rlvr_train_step,
    sample_grouped,
    sft_train_step,
    verifiable_answer_reward,
)
from student_lm import AdamW, GPTConfig, TransformerLM, save_checkpoint  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiny SFT followed by grouped RLVR")
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--rl-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts")
    return parser


def _supervised_batch(
    values: Tensor, *, prompt_token: int, eos_token: int
) -> tuple[Tensor, Tensor]:
    answers = (values + 1) % prompt_token
    tokens = torch.stack(
        [
            torch.full_like(values, prompt_token),
            values,
            answers,
            torch.full_like(values, eos_token),
        ],
        dim=1,
    )
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 2:] = True
    return tokens, mask


def main() -> int:
    args = _build_parser().parse_args()
    if args.sft_steps <= 0 or args.rl_steps <= 0:
        raise ValueError("step counts must be positive")
    torch.manual_seed(336)
    prompt_token, eos_token, vocab_size = 14, 15, 16
    config = GPTConfig(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=8,
    )
    policy = TransformerLM(config)
    optimizer = AdamW(policy.parameters(), lr=2e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(337)
    sft_losses: list[float] = []
    for _ in range(args.sft_steps):
        values = torch.randint(0, prompt_token, (16,), generator=generator)
        tokens, mask = _supervised_batch(values, prompt_token=prompt_token, eos_token=eos_token)
        sft_losses.append(float(sft_train_step(policy, tokens, mask, optimizer)))

    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    rl_metrics: list[dict[str, float]] = []
    for _ in range(args.rl_steps):
        values = torch.randint(0, prompt_token, (8,), generator=generator)
        prompts = torch.stack([torch.full_like(values, prompt_token), values], dim=1)
        rollout = sample_grouped(
            policy,
            prompts,
            group_size=4,
            max_new_tokens=2,
            temperature=1.0,
            generator=generator,
            eos_token_id=eos_token,
        )
        rewards = verifiable_answer_reward(
            rollout.sequences, rollout.response_mask, (values + 1) % prompt_token
        )
        metrics = rlvr_train_step(policy, reference, rollout, rewards, optimizer)
        rl_metrics.append(asdict(metrics))

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "policy.pt"
    save_checkpoint(
        checkpoint,
        policy,
        optimizer,
        step=args.sft_steps + args.rl_steps,
        generator=generator,
        metadata={"config": asdict(config), "task": "modular-successor"},
    )
    report = {
        "sft_first_loss": sft_losses[0],
        "sft_last_loss": sft_losses[-1],
        "rl_metrics": rl_metrics,
        "boundary": "Toy exact-match RLVR is not a substitute for human preference or safety evaluation.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
