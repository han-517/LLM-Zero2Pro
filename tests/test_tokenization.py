import pytest

from llm_from_scratch.tokenization import BytePairTokenizer


@pytest.mark.parametrize("text", ["你好，LLM！", "banana bandana", "", "🙂🙂a"])
def test_byte_bpe_round_trip(text: str) -> None:
    tokenizer = BytePairTokenizer.train("你好，LLM！ banana bandana 🙂🙂a", vocab_size=280)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_is_deterministic_and_serializable() -> None:
    first = BytePairTokenizer.train("abababab cdcd", vocab_size=265)
    second = BytePairTokenizer.train("abababab cdcd", vocab_size=265)
    assert first.to_dict() == second.to_dict()
    restored = BytePairTokenizer.from_dict(first.to_dict())
    assert restored.decode(restored.encode("ab cd")) == "ab cd"


def test_unknown_token_fails_loudly() -> None:
    with pytest.raises(ValueError, match="未知 token"):
        BytePairTokenizer().decode([999])

