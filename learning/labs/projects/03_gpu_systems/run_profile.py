from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT01_ROOT = PROJECT_ROOT.parent / "01_end_to_end_lm"
if str(PROJECT01_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT01_ROOT))

from student_lm import AdamW, GPTConfig, TransformerLM  # noqa: E402
from student_systems import benchmark_callable, ddp_train_step, profile_callable  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a learner-owned Transformer step")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "profile.json")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(336)
    config = GPTConfig(vocab_size=128, d_model=64, num_heads=4, num_layers=2, d_ff=176)
    model = TransformerLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, config.vocab_size, (8, 64), device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    def step() -> object:
        return ddp_train_step(model, inputs, targets, optimizer)

    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    timing = benchmark_callable(
        step, warmup=3, iterations=10, work_items=inputs.numel(), synchronize=synchronize
    )
    operators = profile_callable(step, warmup=1, active_steps=3)
    report = {
        "device": str(device),
        "timing": timing.__dict__,
        "top_operators_self_time_us": sorted(
            operators.items(), key=lambda item: item[1], reverse=True
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
