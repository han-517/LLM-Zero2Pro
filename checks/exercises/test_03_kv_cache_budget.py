from checks.exercises._loader import load_starter

student = load_starter("03_kv_cache_budget.py")


def test_projection_count_distinguishes_prefill_and_recomputation() -> None:
    assert student.projection_token_count(32, 16, use_cache=True) == 47
    assert student.projection_token_count(32, 16, use_cache=False) == sum(range(32, 48))
    assert student.projection_token_count(1, 1, use_cache=True) == 1
    assert student.projection_token_count(1, 1, use_cache=False) == 1


def test_kv_storage_counts_both_key_and_value() -> None:
    assert student.kv_cache_elements(12, 128, 4, 64) == 2 * 12 * 128 * 4 * 64
    assert student.kv_cache_elements(1, 7, 1, 8) == 112
