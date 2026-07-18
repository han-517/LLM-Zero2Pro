import math

from exercises.checks._loader import load_starter

student = load_starter("01_stable_softmax.py")


def test_extreme_logits_are_finite_probabilities() -> None:
    probabilities = student.stable_softmax([10_000.0, 10_001.0, 10_002.0])
    assert len(probabilities) == 3
    assert all(math.isfinite(value) and 0.0 < value < 1.0 for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_softmax_is_translation_invariant() -> None:
    original = student.stable_softmax([-2.0, 0.5, 3.0])
    shifted = student.stable_softmax([value - 9000.0 for value in (-2.0, 0.5, 3.0)])
    for left, right in zip(original, shifted, strict=True):
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def test_equal_logits_produce_uniform_distribution() -> None:
    assert student.stable_softmax([7.0, 7.0, 7.0, 7.0]) == [0.25] * 4
