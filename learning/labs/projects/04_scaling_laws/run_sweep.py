from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT01_ROOT = PROJECT_ROOT.parent / "01_end_to_end_lm"
if str(PROJECT01_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT01_ROOT))

from student_lm import AdamW, GPTConfig, TransformerLM, save_checkpoint  # noqa: E402
from student_lm.training import train_steps  # noqa: E402
from student_scaling import (  # noqa: E402
    ModelSpec,
    RunRecord,
    build_run_grid,
    fit_power_law,
    pareto_frontier,
    validate_run_records,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real tiny-model scaling sweep")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--max-flops", type=float, default=2.0e10)
    return parser


def _config_hash(config: GPTConfig, run_id: str, seed: int) -> str:
    payload = {"config": asdict(config), "run_id": run_id, "seed": seed}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def main() -> int:
    args = _build_parser().parse_args()
    models = [
        ModelSpec("tiny-16", 32, 16, 1, 2, 32),
        ModelSpec("tiny-24", 32, 24, 1, 3, 48),
        ModelSpec("tiny-32", 32, 32, 2, 4, 64),
    ]
    runs = build_run_grid(
        models,
        [2_048, 4_096],
        seeds=(336,),
        batch_size=4,
        sequence_length=16,
        max_flops=args.max_flops,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    records: list[RunRecord] = []
    tokens = torch.tensor(([index % 32 for index in range(20_000)]), dtype=torch.long)

    for run in runs:
        torch.manual_seed(run.seed)
        config = GPTConfig(
            vocab_size=run.model.vocab_size,
            d_model=run.model.d_model,
            num_heads=run.model.num_heads,
            num_layers=run.model.num_layers,
            d_ff=run.model.d_ff,
            max_seq_len=run.sequence_length,
        )
        model = TransformerLM(config)
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != run.parameter_count:
            raise RuntimeError("parameter-count estimator disagrees with the instantiated model")
        optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        generator = torch.Generator().manual_seed(run.seed + 1)
        started = time.perf_counter()
        losses = train_steps(
            model,
            tokens,
            optimizer,
            steps=run.steps,
            batch_size=run.batch_size,
            sequence_length=run.sequence_length,
            generator=generator,
        )
        elapsed = time.perf_counter() - started
        checkpoint = checkpoint_dir / f"{run.run_id}.pt"
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            step=run.steps,
            generator=generator,
            metadata={"run_spec": asdict(run)},
        )
        records.append(
            RunRecord(
                run_id=run.run_id,
                model_name=run.model.name,
                parameter_count=actual_parameters,
                training_tokens=run.training_tokens,
                predicted_flops=run.predicted_flops,
                initial_loss=losses[0],
                final_loss=losses[-1],
                elapsed_seconds=elapsed,
                steps=run.steps,
                seed=run.seed,
                config_sha256=_config_hash(config, run.run_id, run.seed),
                checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
            )
        )

    validate_run_records(records)
    lines = [json.dumps(asdict(record), sort_keys=True) for record in records]
    (args.output / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frontier = pareto_frontier(records)
    fit = fit_power_law(
        [record.predicted_flops for record in frontier],
        [record.final_loss for record in frontier],
    )
    summary = {
        "run_count": len(records),
        "total_predicted_flops": sum(record.predicted_flops for record in records),
        "frontier_run_ids": [record.run_id for record in frontier],
        "fit": asdict(fit),
        "warning": "The 6ND proxy is not measured accelerator FLOPs; tiny runs are not an LLM forecast.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
