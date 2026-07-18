import pytest
import torch

from llm_from_scratch.inference import (
    PageTable,
    RequestTrace,
    greedy_speculative_decode,
    stochastic_speculative_decode,
    summarize_serving,
    symmetric_dequantize,
    symmetric_quantize,
)


def test_symmetric_quantization_zero_error_order_and_validation() -> None:
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
    with pytest.raises(ValueError, match="有限"):
        symmetric_quantize(torch.tensor([float("nan")]))


def test_page_table_allocates_maps_releases_and_reports_fragmentation() -> None:
    table = PageTable(page_size=4, free_pages=list(range(5)))
    assert table.append_tokens("a", 5) == [0, 1]
    assert table.logical_to_physical("a", 4) == (1, 0)
    assert table.append_tokens("b", 4) == [2]
    assert table.internal_fragmentation_tokens == 3
    table.release("a")
    assert table.free_pages == [0, 1, 3, 4]
    assert table.allocated_pages == 1


def test_page_table_prefix_sharing_refcounts_and_copy_on_write() -> None:
    table = PageTable(page_size=4, free_pages=list(range(5)))
    table.append_tokens("source", 5)
    assert table.share_prefix("source", "branch") == [0, 1]
    assert table.page_refcounts == {0: 2, 1: 2}

    # branch 的最后一页是共享 partial page，追加前必须 COW。
    assert table.append_tokens("branch", 1) == [0, 2]
    assert table.logical_to_physical("source", 4) == (1, 0)
    assert table.logical_to_physical("branch", 4) == (2, 0)
    assert table.page_refcounts == {0: 2, 1: 1, 2: 1}
    assert table.internal_fragmentation_tokens == 5
    table.release("source")
    assert table.page_refcounts == {0: 1, 2: 1}
    assert 1 in table.free_pages


def test_page_table_failed_append_is_atomic() -> None:
    table = PageTable(page_size=2, free_pages=[0])
    with pytest.raises(MemoryError):
        table.append_tokens("a", 3)
    assert table.free_pages == [0]
    assert table.sequence_pages["a"] == []
    assert "a" not in table.sequence_lengths


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


def test_stochastic_speculative_decode_accepts_bonus_and_corrects_rejection() -> None:
    generator = torch.Generator().manual_seed(1)

    def draft(_context):
        return torch.tensor([1.0, 0.0])

    def accepting_target(_prefix, candidates):
        return [torch.tensor([1.0, 0.0])] * len(candidates) + [torch.tensor([0.0, 1.0])]

    output, stats = stochastic_speculative_decode(
        draft, accepting_target, [9], 3, draft_steps=2, generator=generator
    )
    assert output == [9, 0, 0, 1]
    assert stats.accepted == 2
    assert stats.target_calls == 1

    def rejecting_target(_prefix, candidates):
        return [torch.tensor([0.0, 1.0])] * (len(candidates) + 1)

    output, stats = stochastic_speculative_decode(
        draft, rejecting_target, [9], 1, generator=generator
    )
    assert output == [9, 1]
    assert stats.accepted == 0


def test_stochastic_speculative_decode_preserves_target_distribution() -> None:
    generator = torch.Generator().manual_seed(123)

    def draft(_context):
        return torch.tensor([0.8, 0.2])

    def target(_prefix, candidates):
        return [torch.tensor([0.3, 0.7])] * (len(candidates) + 1)

    samples = []
    for _ in range(2000):
        output, _ = stochastic_speculative_decode(draft, target, [], 1, generator=generator)
        samples.append(output[0])
    observed_one = sum(samples) / len(samples)
    assert observed_one == pytest.approx(0.7, abs=0.04)


def test_serving_metrics_include_queueing_percentiles_throughput_and_goodput() -> None:
    traces = [
        RequestTrace(0.0, 2.0, 5.0, output_tokens=4, prompt_tokens=6),
        RequestTrace(1.0, 2.0, 4.0, output_tokens=3, prompt_tokens=2),
    ]
    metrics = summarize_serving(traces, ttft_slo=1.5, tpot_slo=1.0)
    assert metrics.ttft.mean == 1.5
    assert metrics.ttft.p50 == 1.5
    assert metrics.tpot is not None and metrics.tpot.mean == 1.0
    assert metrics.request_throughput == 0.4
    assert metrics.output_token_throughput == 1.4
    assert metrics.total_token_throughput == 3.0
    assert metrics.goodput == 0.2


def test_single_output_token_has_no_tpot() -> None:
    metrics = summarize_serving([RequestTrace(0, 1, 2, output_tokens=1)])
    assert metrics.tpot is None
