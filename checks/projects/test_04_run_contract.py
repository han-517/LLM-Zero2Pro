from __future__ import annotations

from dataclasses import replace

import pytest
from student_scaling import RunRecord, pareto_frontier, validate_run_records


def _record(run_id: str, flops: float, loss: float, *, seed: int = 1) -> RunRecord:
    parameters = 100
    tokens = int(flops / (6 * parameters))
    return RunRecord(
        run_id=run_id,
        model_name="tiny",
        parameter_count=parameters,
        training_tokens=tokens,
        predicted_flops=float(6 * parameters * tokens),
        initial_loss=4.0,
        final_loss=loss,
        elapsed_seconds=0.1,
        steps=2,
        seed=seed,
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )


def test_pareto_frontier_removes_compute_loss_dominated_runs() -> None:
    records = [
        _record("a", 6000, 3.0),
        _record("b", 12000, 2.7),
        _record("dominated", 18000, 2.9),
        _record("c", 24000, 2.5),
    ]
    assert [record.run_id for record in pareto_frontier(records)] == ["a", "b", "c"]


def test_run_record_validation_requires_reproducible_real_run_fields() -> None:
    valid = [_record("a", 6000, 3.0), _record("b", 12000, 2.7, seed=2)]
    validate_run_records(valid)
    with pytest.raises(ValueError):
        validate_run_records([valid[0], replace(valid[1], run_id="a")])
    with pytest.raises(ValueError):
        validate_run_records([replace(valid[0], checkpoint_sha256="missing")])
    with pytest.raises(ValueError):
        validate_run_records([replace(valid[0], predicted_flops=123.0)])
    with pytest.raises(ValueError):
        validate_run_records([replace(valid[0], final_loss=float("nan"))])
