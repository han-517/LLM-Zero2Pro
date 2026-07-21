"""Project 01 byte-level BPE tokenizer starter.

Core algorithms are intentionally blank. Keep the implementation deterministic and do not
cross document boundaries while counting pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

Merge = tuple[int, int, int]


class ByteBPETokenizer:
    def __init__(
        self,
        vocab: Mapping[int, bytes],
        merges: Sequence[Merge],
        special_tokens: Mapping[str, int] | None = None,
    ) -> None:
        self.vocab = {int(index): bytes(piece) for index, piece in vocab.items()}
        self.merges = tuple((int(a), int(b), int(c)) for a, b, c in merges)
        self.special_tokens = dict(special_tokens or {})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        documents: Sequence[str],
        vocab_size: int,
        special_tokens: Sequence[str] = (),
    ) -> ByteBPETokenizer:
        """Train deterministic byte BPE without merging across documents/special tokens."""

        # TODO: initialize all 256 byte tokens, split special tokens, count adjacent pairs,
        # choose a deterministic winner, merge every non-overlapping occurrence, and store
        # the bytes represented by each new token.
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode arbitrary Unicode text; configured special tokens must stay atomic."""

        # TODO: UTF-8 encode ordinary spans, then apply learned merges in training order.
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode IDs using strict vocabulary validation and UTF-8 replacement semantics."""

        # TODO: concatenate token bytes and decode once; preserve configured special tokens.
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write a versioned JSON representation with byte pieces encoded losslessly."""

        # TODO: make the file deterministic so equal tokenizers produce equal bytes.
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        """Load and validate a file written by :meth:`save`."""

        # TODO: reject unsupported versions, duplicate IDs and malformed merges.
        raise NotImplementedError
