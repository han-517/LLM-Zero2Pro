from __future__ import annotations

from hashlib import sha256

from student_pipeline import (
    AuditEvent,
    PipelineConfig,
    RawDocument,
    build_data_card,
    detect_language,
    extract_visible_text,
    process_documents,
    quality_metrics,
    redact_sensitive,
)


def test_html_extraction_keeps_visible_text_only() -> None:
    html = "<style>.secret{}</style><h1>A &amp; B</h1><script>hidden()</script><p>Useful text.</p>"
    text = extract_visible_text(html)
    assert text == "A & B Useful text."
    assert "secret" not in text and "hidden" not in text


def test_toy_language_detector_has_an_unknown_state() -> None:
    assert detect_language("A careful language model uses prior context.")[0] == "en"
    assert detect_language("语言模型需要可靠的数据与评测。 ")[0] == "zh"
    assert detect_language("1234 --- !!!")[0] == "unknown"
    assert all(0.0 <= detect_language(text)[1] <= 1.0 for text in ("hello", "语言", "123"))


def test_redaction_removes_literals_and_does_not_log_them() -> None:
    source = "mail me at person@example.org; api_key = sk-test-1234567890abcdef"
    redacted, findings = redact_sensitive(source)
    assert "person@example.org" not in redacted
    assert "sk-test-1234567890abcdef" not in redacted
    assert {finding.kind for finding in findings} == {"email", "api_key"}
    for finding in findings:
        assert len(finding.digest) == 64
        int(finding.digest, 16)
        assert "person" not in repr(finding) and "sk-test" not in repr(finding)


def test_quality_metrics_are_explicit_for_empty_and_repeated_text() -> None:
    empty = quality_metrics("")
    assert empty == {
        "character_count": 0.0,
        "alphabetic_ratio": 0.0,
        "unique_line_ratio": 0.0,
        "mean_word_length": 0.0,
    }
    repeated = quality_metrics("same line\nsame line\ndifferent line")
    assert repeated["character_count"] > 0
    assert repeated["unique_line_ratio"] == 2 / 3


def test_pipeline_and_data_card_reconcile_every_input() -> None:
    long_text = (
        "A reproducible language-model dataset records sources, rules, hashes, and decisions."
    )
    records = [
        RawDocument("a", "https://a", "2026-01-01", "allowed", f"<p>{long_text}</p>"),
        RawDocument("b", "https://b", "2026-01-01", "unknown", f"<p>{long_text}</p>"),
    ]
    config = PipelineConfig(min_characters=20, min_alphabetic_ratio=0.5)
    processed, events = process_documents(records, config)
    terminal = [event for event in events if event.stage == "final"]
    assert len(terminal) == len(records)
    assert {event.document_id for event in terminal} == {"a", "b"}
    assert {event.decision for event in terminal} == {"accepted", "rejected"}
    assert all(isinstance(event, AuditEvent) for event in events)

    card = build_data_card(records, processed, events)
    assert card["raw_documents"] == 2
    assert card["accepted_documents"] + card["rejected_documents"] == 2
    assert card["raw_characters"] >= card["accepted_characters"]
    assert sha256(records[0].html.encode()).hexdigest() in {
        event.input_sha256 for event in events if event.document_id == "a"
    }
