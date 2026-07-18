import pytest

from exercises.checks._loader import load_starter

student = load_starter("14_data_pipeline.py")


def test_dedup_is_stable_and_reports_origin() -> None:
    docs, indices = student.exact_deduplicate(["a", "b", "a", ""])
    assert docs == ["a", "b", ""]
    assert indices == [0, 1, 3]


def test_packing_preserves_order_and_boundaries() -> None:
    assert student.pack_sequences([[1, 2], [3]], 4, 9) == [[1, 2, 9, 3], [9]]


def test_invalid_block_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        student.pack_sequences([[1]], 0, 9)
