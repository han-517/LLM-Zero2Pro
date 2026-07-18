from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DocumentSplit:
    """先按文档切分，再分别构造重叠窗口，避免相邻窗口泄漏。"""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def split_documents(
    documents: Sequence[str],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 20260718,
) -> DocumentSplit:
    """确定性地打乱并按完整文档切分；每个 split 至少包含一篇文档。"""

    if len(documents) < 3:
        raise ValueError("至少需要 3 篇文档，才能建立 train/validation/test")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction 必须在 (0, 1) 内")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 (0, 1) 内")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction 必须小于 1")
    if any(not isinstance(document, str) for document in documents):
        raise TypeError("documents 中的每一项都必须是字符串")

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = min(max(1, int(count * train_fraction)), count - 2)
    validation_count = min(max(1, int(count * validation_fraction)), count - train_count - 1)
    validation_end = train_count + validation_count
    return DocumentSplit(
        train=tuple(shuffled[:train_count]),
        validation=tuple(shuffled[train_count:validation_end]),
        test=tuple(shuffled[validation_end:]),
    )


def make_next_token_windows(
    token_ids: Tensor, context_size: int, *, stride: int = 1
) -> tuple[Tensor, Tensor]:
    """把一篇文档变为 ``contexts:[N,C]`` 和 ``targets:[N]``。"""

    if token_ids.ndim != 1:
        raise ValueError("token_ids 必须是一维")
    if context_size < 1:
        raise ValueError("context_size 必须 >= 1")
    if stride < 1:
        raise ValueError("stride 必须 >= 1")
    if token_ids.numel() <= context_size:
        raise ValueError("token 数必须大于 context_size")

    starts = range(0, token_ids.numel() - context_size, stride)
    contexts = torch.stack([token_ids[start : start + context_size] for start in starts])
    targets = torch.stack([token_ids[start + context_size] for start in starts])
    return contexts, targets


def make_document_windows(
    token_documents: Sequence[Tensor], context_size: int, *, stride: int = 1
) -> tuple[Tensor, Tensor]:
    """逐文档造窗口再拼接，绝不让一个样本跨越两篇文档。"""

    batches: list[tuple[Tensor, Tensor]] = []
    for token_ids in token_documents:
        if token_ids.ndim != 1:
            raise ValueError("每篇 token 文档都必须是一维")
        if token_ids.numel() > context_size:
            batches.append(make_next_token_windows(token_ids, context_size, stride=stride))
    if not batches:
        raise ValueError("没有任何文档长到足以构造一个窗口")
    contexts, targets = zip(*batches, strict=True)
    return torch.cat(contexts), torch.cat(targets)


def sample_from_logits(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """从 ``[B,V]`` logits 采样；temperature=0 表示贪心。"""

    if logits.ndim != 2:
        raise ValueError("logits 必须是 [B,V]")
    if temperature < 0:
        raise ValueError("temperature 不能为负数")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


class BigramLanguageModel(nn.Module):
    """当前 token 直接查表得到下一个 token 的 logits。"""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size 必须 >= 2")
        self.vocab_size = vocab_size
        self.transition_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.transition_logits(token_ids)
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 必须与 token_ids 同形状")
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _validate_prefix(prefix, max_new_tokens)
        output = prefix
        for _ in range(max_new_tokens):
            logits, _ = self(output[:, -1:])
            next_token = sample_from_logits(
                logits[:, -1], temperature=temperature, generator=generator
            )
            output = torch.cat((output, next_token), dim=1)
        return output


class FixedWindowMLP(nn.Module):
    """拼接最近 ``context_size`` 个 embedding 后预测一个 token。"""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(vocab_size - 1, context_size, embedding_dim, hidden_dim) < 1:
            raise ValueError("vocab_size 必须 >= 2，其他维度必须为正数")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, context_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if context_ids.ndim != 2 or context_ids.shape[1] != self.context_size:
            raise ValueError(f"context_ids 必须是 [B,{self.context_size}]")
        embedded = self.embedding(context_ids).flatten(start_dim=1)
        logits = self.output(torch.tanh(self.hidden(embedded)))
        loss = None
        if targets is not None:
            if targets.shape != (context_ids.shape[0],):
                raise ValueError("targets 必须是 [B]")
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _validate_prefix(prefix, max_new_tokens)
        if prefix.shape[1] < self.context_size:
            raise ValueError("prefix 长度不能小于 context_size")
        output = prefix
        for _ in range(max_new_tokens):
            logits, _ = self(output[:, -self.context_size :])
            next_token = sample_from_logits(logits, temperature=temperature, generator=generator)
            output = torch.cat((output, next_token), dim=1)
        return output


class ElmanRNNLanguageModel(nn.Module):
    """显式时间循环的 Elman RNN，用于观察隐藏状态与串行瓶颈。"""

    def __init__(self, vocab_size: int, embedding_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        if min(vocab_size - 1, embedding_dim, hidden_dim) < 1:
            raise ValueError("vocab_size 必须 >= 2，其他维度必须为正数")
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_to_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_to_logits = nn.Linear(hidden_dim, vocab_size)

    def step(self, token_ids: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """执行一个时间步；输入 ``[B]`` token 与 ``[B,H]`` hidden。"""

        if token_ids.ndim != 1:
            raise ValueError("单步 token_ids 必须是 [B]")
        if hidden.shape != (token_ids.shape[0], self.hidden_dim):
            raise ValueError(f"hidden 必须是 [B,{self.hidden_dim}]")
        embedded = self.embedding(token_ids)
        next_hidden = torch.tanh(self.input_to_hidden(embedded) + self.hidden_to_hidden(hidden))
        return self.hidden_to_logits(next_hidden), next_hidden

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids 必须是非空 [B,T]")
        batch, time = token_ids.shape
        if hidden is None:
            hidden = torch.zeros(
                batch,
                self.hidden_dim,
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
        elif hidden.shape != (batch, self.hidden_dim):
            raise ValueError(f"hidden 必须是 [B,{self.hidden_dim}]")

        pieces: list[Tensor] = []
        for index in range(time):
            current_logits, hidden = self.step(token_ids[:, index], hidden)
            pieces.append(current_logits)
        logits = torch.stack(pieces, dim=1)
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 必须与 token_ids 同形状")
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss, hidden

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _validate_prefix(prefix, max_new_tokens)
        output = prefix
        logits, _, hidden = self(prefix)
        next_logits = logits[:, -1]
        for step_index in range(max_new_tokens):
            next_token = sample_from_logits(
                next_logits, temperature=temperature, generator=generator
            )
            output = torch.cat((output, next_token), dim=1)
            if step_index + 1 < max_new_tokens:
                next_logits, hidden = self.step(next_token[:, 0], hidden)
        return output


def _validate_prefix(prefix: Tensor, max_new_tokens: int) -> None:
    if prefix.ndim != 2 or prefix.shape[1] == 0:
        raise ValueError("prefix 必须是非空 [B,T]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")
