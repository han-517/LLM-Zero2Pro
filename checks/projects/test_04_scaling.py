from __future__ import annotations

import math

import pytest
from student_scaling import (
    ModelSpec,
    bootstrap_exponent,
    build_run_grid,
    decoder_parameter_count,
    estimate_training_flops,
    fit_power_law,
)


def test_parameter_count_matches_project_decoder_contract() -> None:
    model = ModelSpec("small", vocab_size=101, d_model=16, num_layers=2, num_heads=4, d_ff=40)
    expected = 101 * 16 + 2 * (4 * 16 * 16 + 3 * 16 * 40 + 2 * 16) + 16
    assert decoder_parameter_count(model) == expected
    assert decoder_parameter_count(model, tied_embeddings=False) == expected + 101 * 16


def test_run_grid_rounds_steps_and_enforces_total_budget() -> None:
    models = [ModelSpec("a", 32, 16, 1, 2, 32), ModelSpec("b", 32, 24, 1, 3, 48)]
    grid = build_run_grid(
        models,
        [1_000, 2_000],
        seeds=(3, 5),
        batch_size=4,
        sequence_length=16,
        max_flops=1e12,
    )
    assert len(grid) == 8
    assert len({run.run_id for run in grid}) == len(grid)
    assert all(run.training_tokens == run.steps * 4 * 16 for run in grid)
    assert all(run.training_tokens >= run.requested_tokens for run in grid)
    assert sum(run.predicted_flops for run in grid) <= 1e12
    with pytest.raises(ValueError):
        build_run_grid(
            models,
            [10_000],
            seeds=(1,),
            batch_size=4,
            sequence_length=16,
            max_flops=1.0,
        )


def test_power_law_fit_recovers_known_exponent_and_reports_residuals() -> None:
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    y = [0.7 + 3.5 * value**-0.4 for value in x]
    fit = fit_power_law(x, y, asymptote=0.7)
    assert math.isclose(fit.coefficient, 3.5, rel_tol=1e-10)
    assert math.isclose(fit.exponent, -0.4, rel_tol=1e-10)
    assert fit.r_squared > 0.999999
    assert len(fit.residuals) == len(x)


def test_bootstrap_is_seeded_and_contains_the_observed_exponent() -> None:
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    y = [
        2.0 * value**-0.25 * noise
        for value, noise in zip(x, [1.02, 0.98, 1.01, 0.99, 1.0, 1.02, 0.98], strict=True)
    ]
    interval = bootstrap_exponent(x, y, samples=200, seed=11)
    assert interval == bootstrap_exponent(x, y, samples=200, seed=11)
    assert interval[0] <= -0.25 <= interval[2]
    assert estimate_training_flops(1_000, 2_000) == 12_000_000.0
