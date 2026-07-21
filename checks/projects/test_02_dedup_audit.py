from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import student_pipeline
from student_pipeline import (
    estimated_jaccard,
    minhash_signature,
    near_duplicate_groups,
    stable_group_split,
)


def test_minhash_is_stable_across_python_processes() -> None:
    text = "language models need auditable data pipelines and reproducible evaluation"
    local = minhash_signature(text, permutations=32, seed=7)
    code = (
        "from student_pipeline import minhash_signature; "
        f"print(','.join(map(str, minhash_signature({text!r}, permutations=32, seed=7))))"
    )
    project_root = Path(student_pipeline.__file__).resolve().parent.parent
    inherited = os.environ.get("PYTHONPATH", "")
    python_path = str(project_root) + (os.pathsep + inherited if inherited else "")
    remote = tuple(
        map(
            int,
            subprocess.check_output(
                [sys.executable, "-c", code],
                text=True,
                env={**os.environ, "PYTHONPATH": python_path},
            ).split(","),
        )
    )
    assert local == remote
    assert len(local) == 32


def test_near_duplicate_grouping_is_transitive_and_deterministic() -> None:
    base = "a careful dataset pipeline records every source hash rule and filtering decision"
    documents = {
        "c": base + " for review",
        "a": base,
        "b": base + " for reviewers",
        "z": "mixture of experts routing is a completely unrelated document",
    }
    groups = near_duplicate_groups(documents, threshold=0.55, permutations=128, seed=9)
    assert groups["a"] == groups["b"] == groups["c"] == "a"
    assert groups["z"] == "z"
    assert groups == near_duplicate_groups(documents, threshold=0.55, permutations=128, seed=9)


def test_signature_similarity_and_split_boundaries() -> None:
    left = minhash_signature("alpha beta gamma delta epsilon", permutations=64)
    right = minhash_signature("alpha beta gamma delta epsilon", permutations=64)
    assert estimated_jaccard(left, right) == 1.0
    assert stable_group_split("duplicate-family-1", seed=2) == stable_group_split(
        "duplicate-family-1", seed=2
    )
    with pytest.raises(ValueError):
        estimated_jaccard(left, right[:-1])
    with pytest.raises(ValueError):
        stable_group_split("x", validation_fraction=1.0)
