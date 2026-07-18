from collections import Counter

from exercises.checks._loader import load_starter

student = load_starter("06_byte_bpe.py")


def test_pair_count_includes_overlapping_occurrences() -> None:
    assert student.count_adjacent_pairs([1, 1, 1]) == Counter({(1, 1): 2})
    assert student.count_adjacent_pairs([1]) == Counter()


def test_pair_choice_is_deterministic_on_ties() -> None:
    counts = Counter({(7, 9): 3, (2, 8): 3, (1, 99): 1})
    assert student.choose_pair(counts) == (2, 8)


def test_merge_is_left_to_right_non_overlapping_and_pure() -> None:
    sequence = [1, 1, 1, 2, 1, 1]
    assert student.merge_pair(sequence, (1, 1), 256) == [256, 1, 2, 256]
    assert sequence == [1, 1, 1, 2, 1, 1]
