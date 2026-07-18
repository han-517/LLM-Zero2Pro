import torch

from llm_from_scratch.inference import (
    PageTable,
    greedy_speculative_decode,
    symmetric_dequantize,
    symmetric_quantize,
)


def test_symmetric_quantization_zero_and_error_order() -> None:
    zeros = torch.zeros(5)
    q_zero, scale_zero = symmetric_quantize(zeros, 4)
    assert torch.count_nonzero(q_zero) == 0
    assert scale_zero == 1

    x = torch.linspace(-1, 1, 101)
    q8, s8 = symmetric_quantize(x, 8)
    q4, s4 = symmetric_quantize(x, 4)
    error8 = (x - symmetric_dequantize(q8, s8)).abs().mean()
    error4 = (x - symmetric_dequantize(q4, s4)).abs().mean()
    assert error8 < error4


def test_page_table_allocates_and_releases() -> None:
    table = PageTable(page_size=4, free_pages=list(range(4)))
    assert table.append_tokens("a", 5) == [0, 1]
    assert table.append_tokens("b", 4) == [2]
    table.release("a")
    assert table.free_pages == [0, 1, 3]


def test_greedy_speculative_decode_accepts_or_falls_back() -> None:
    def target(prefix):
        return (prefix[-1] + 1) % 10

    verifier_calls = 0

    def target_verify(prefix, candidates):
        nonlocal verifier_calls
        verifier_calls += 1
        context = list(prefix)
        verified = []
        for candidate in candidates:
            verified.append(target(context))
            context.append(candidate)
        return verified

    def same_draft(prefix):
        return (prefix[-1] + 1) % 10

    output, stats = greedy_speculative_decode(same_draft, target_verify, [0], 6, draft_steps=3)
    assert output == [0, 1, 2, 3, 4, 5, 6]
    assert stats.accepted == 6
    assert stats.target_calls == verifier_calls == 2

    def bad_draft(prefix):
        return 9

    verifier_calls = 0
    output, stats = greedy_speculative_decode(bad_draft, target_verify, [0], 3, draft_steps=3)
    assert output == [0, 1, 2, 3]
    assert stats.accepted == 0
    assert stats.target_calls == verifier_calls == 3

