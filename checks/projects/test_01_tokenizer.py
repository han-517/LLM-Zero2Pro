from __future__ import annotations

from student_lm.tokenizer import ByteBPETokenizer


def _train() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["banana bandana", "你好，language model", "do not cross documents"],
        vocab_size=272,
        special_tokens=("<eos>",),
    )


def test_byte_bpe_is_deterministic_and_round_trips_unicode() -> None:
    first = _train()
    second = _train()
    assert first.vocab == second.vocab
    assert first.merges == second.merges
    for text in ("", "banana", "未登录词🙂", "a<eos>b"):
        assert first.decode(first.encode(text)) == text


def test_special_token_stays_atomic() -> None:
    tokenizer = _train()
    eos_id = tokenizer.special_tokens["<eos>"]
    encoded = tokenizer.encode("left<eos>right")
    assert encoded.count(eos_id) == 1
    assert tokenizer.decode(encoded) == "left<eos>right"


def test_tokenizer_serialization_is_lossless(tmp_path) -> None:
    tokenizer = _train()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = ByteBPETokenizer.load(path)
    assert restored.vocab == tokenizer.vocab
    assert restored.merges == tokenizer.merges
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.decode(restored.encode("reload 后仍然可逆")) == "reload 后仍然可逆"
