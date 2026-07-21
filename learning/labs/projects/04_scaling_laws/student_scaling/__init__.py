"""Learner-owned package for project 04."""

from .scaling import (
    ModelSpec,
    PowerLawFit,
    RunRecord,
    RunSpec,
    bootstrap_exponent,
    build_run_grid,
    decoder_parameter_count,
    estimate_training_flops,
    fit_power_law,
    pareto_frontier,
    validate_run_records,
)

__all__ = [
    "ModelSpec",
    "PowerLawFit",
    "RunRecord",
    "RunSpec",
    "bootstrap_exponent",
    "build_run_grid",
    "decoder_parameter_count",
    "estimate_training_flops",
    "fit_power_law",
    "pareto_frontier",
    "validate_run_records",
]
