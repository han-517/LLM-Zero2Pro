from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Pair = tuple[int, int]


@dataclass(frozen=True)
class Merge:
    pair: Pair
    token_id: int


def _merge_pair(sequence: list[int], pair: Pair, new_id: int) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


class BytePairTokenizer:
    """确定性的教学版 Byte-level BPE。

    初始 0..255 对应单字节。训练时出现次数并列的 pair 按整数对排序，
    使相同语料和参数总能得到相同 merge 表。
    """

    def __init__(self, merges: list[Merge] | None = None) -> None:
        self.merges = list(merges or [])
        self._token_bytes: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for merge in self.merges:
            left, right = merge.pair
            self._token_bytes[merge.token_id] = self._token_bytes[left] + self._token_bytes[right]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 300) -> BytePairTokenizer:
        if vocab_size < 256:
            raise ValueError("Byte-level BPE 的 vocab_size 不能小于 256")
        sequence = list(text.encode("utf-8"))
        merges: list[Merge] = []
        next_id = 256
        while next_id < vocab_size and len(sequence) >= 2:
            counts = Counter(zip(sequence, sequence[1:], strict=False))
            if not counts:
                break
            best_count = max(counts.values())
            best_pair = min(pair for pair, count in counts.items() if count == best_count)
            merges.append(Merge(pair=best_pair, token_id=next_id))
            sequence = _merge_pair(sequence, best_pair, next_id)
            next_id += 1
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        if len(sequence) < 2:
            return sequence
        ranks = {merge.pair: index for index, merge in enumerate(self.merges)}
        ids = {merge.pair: merge.token_id for merge in self.merges}
        while len(sequence) >= 2:
            present = set(zip(sequence, sequence[1:], strict=False))
            candidates = [pair for pair in present if pair in ranks]
            if not candidates:
                break
            best_pair = min(candidates, key=ranks.__getitem__)
            sequence = _merge_pair(sequence, best_pair, ids[best_pair])
        return sequence

    def decode_bytes(self, token_ids: list[int]) -> bytes:
        try:
            return b"".join(self._token_bytes[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"未知 token id: {exc.args[0]}") from exc

    def decode(self, token_ids: list[int], errors: str = "strict") -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def to_dict(self) -> dict:
        return {
            "type": "byte_bpe",
            "merges": [
                {"left": merge.pair[0], "right": merge.pair[1], "token_id": merge.token_id}
                for merge in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BytePairTokenizer:
        if data.get("type") != "byte_bpe":
            raise ValueError("不是受支持的 byte_bpe tokenizer")
        merges = [
            Merge(pair=(int(item["left"]), int(item["right"])), token_id=int(item["token_id"]))
            for item in data.get("merges", [])
        ]
        expected_ids = list(range(256, 256 + len(merges)))
        if [merge.token_id for merge in merges] != expected_ids:
            raise ValueError("merge token_id 必须从 256 连续递增")
        return cls(merges)

