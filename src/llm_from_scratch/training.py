from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def seed_everything(seed: int = 20260718) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exact_deduplicate(documents: Sequence[str]) -> tuple[list[str], list[int]]:
    """保留第一次出现的完全相同文档，并返回被移除文档的原始下标。

    这里故意不做小写化或空白归一化：那会把“精确去重”变成另一条需要单独
    评估误删率的规范化规则。
    """

    seen: set[str] = set()
    unique: list[str] = []
    duplicate_indices: list[int] = []
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            raise TypeError("documents 中的每一项都必须是字符串")
        if document in seen:
            duplicate_indices.append(index)
        else:
            seen.add(document)
            unique.append(document)
    return unique, duplicate_indices


def _word_ngrams(text: str, ngram_size: int, *, lowercase: bool) -> set[tuple[str, ...]]:
    words = text.casefold().split() if lowercase else text.split()
    return {
        tuple(words[start : start + ngram_size])
        for start in range(max(0, len(words) - ngram_size + 1))
    }


def detect_ngram_contamination(
    training_documents: Sequence[str],
    evaluation_documents: Sequence[str],
    *,
    ngram_size: int = 8,
    lowercase: bool = True,
) -> list[bool]:
    """用词级 n-gram 重叠标记可能污染评测集的训练文档。

    这是高召回的教学筛查器，不是“证明数据泄漏”的判决器。短于 ``ngram_size``
    的文档不会产生指纹；真实管道还应记录命中片段并进行人工抽检。
    """

    if ngram_size < 1:
        raise ValueError("ngram_size 必须 >= 1")
    for document in [*training_documents, *evaluation_documents]:
        if not isinstance(document, str):
            raise TypeError("训练和评测文档都必须是字符串")
    evaluation_ngrams: set[tuple[str, ...]] = set()
    for document in evaluation_documents:
        evaluation_ngrams.update(_word_ngrams(document, ngram_size, lowercase=lowercase))
    return [
        bool(_word_ngrams(document, ngram_size, lowercase=lowercase) & evaluation_ngrams)
        for document in training_documents
    ]


def deterministic_mixture_sample[T](
    sources: Mapping[str, Sequence[T]],
    weights: Mapping[str, float],
    total_samples: int,
    *,
    seed: int,
) -> list[tuple[str, T]]:
    """按给定权重确定性抽取带来源标签的样本。

    采样有放回，适合演示数据混合，而不是替代大规模流式数据加载器。返回来源标签
    是为了让调用方可以审计实际混合比例。
    """

    if total_samples < 0:
        raise ValueError("total_samples 不能为负")
    if not sources:
        if total_samples == 0:
            return []
        raise ValueError("sources 不能为空")
    if set(sources) != set(weights):
        raise ValueError("sources 与 weights 必须包含完全相同的来源")
    names = list(sources)
    numeric_weights = []
    for name in names:
        if not sources[name]:
            raise ValueError(f"来源 {name!r} 不能为空")
        weight = float(weights[name])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("混合权重必须是有限非负数")
        numeric_weights.append(weight)
    if sum(numeric_weights) <= 0:
        raise ValueError("至少一个混合权重必须为正")

    generator = random.Random(seed)
    sampled = []
    for _ in range(total_samples):
        name = generator.choices(names, weights=numeric_weights, k=1)[0]
        values = sources[name]
        sampled.append((name, values[generator.randrange(len(values))]))
    return sampled


def pack_documents(
    documents: Sequence[Sequence[int] | Tensor],
    *,
    block_size: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    """把 token 文档打包为定长块，返回 ``input_ids`` 与 label-aligned mask。

    每个文档后插入 EOS；每个新文档的首 token 不作为 label，避免跨文档预测。
    ``loss_mask[b, t]`` 表示 ``input_ids[b, t]`` 可以充当监督 label。块首位置在
    因果 shift 后本来也不会参与损失。
    """

    if block_size < 1:
        raise ValueError("block_size 必须 >= 1")
    stream: list[int] = []
    label_mask: list[bool] = []
    for document in documents:
        values = document.tolist() if isinstance(document, Tensor) else list(document)
        if not values:
            raise ValueError("文档 token 序列不能为空")
        if any(not isinstance(token, int) for token in values):
            raise TypeError("文档 token 必须都是整数")
        stream.extend(values)
        stream.append(eos_token_id)
        label_mask.extend([False, *([True] * len(values))])

    if not stream:
        return (
            torch.empty((0, block_size), dtype=torch.long),
            torch.empty((0, block_size), dtype=torch.bool),
        )
    remainder = len(stream) % block_size
    if remainder:
        padding = block_size - remainder
        stream.extend([pad_token_id] * padding)
        label_mask.extend([False] * padding)
    return (
        torch.tensor(stream, dtype=torch.long).reshape(-1, block_size),
        torch.tensor(label_mask, dtype=torch.bool).reshape(-1, block_size),
    )


@torch.no_grad()
def adamw_step_(
    parameter: Tensor,
    gradient: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    """执行一个教学版 AdamW 原地更新，与 PyTorch 的基本单参数公式对齐。"""

    if parameter.shape != gradient.shape or parameter.shape != exp_avg.shape:
        raise ValueError("parameter、gradient 与 exp_avg 必须同形状")
    if parameter.shape != exp_avg_sq.shape:
        raise ValueError("parameter 与 exp_avg_sq 必须同形状")
    if step < 1:
        raise ValueError("step 从 1 开始")
    if learning_rate < 0 or weight_decay < 0 or eps <= 0:
        raise ValueError("learning_rate/weight_decay 必须非负，eps 必须为正")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("beta1 和 beta2 必须位于 [0, 1)")

    parameter.mul_(1 - learning_rate * weight_decay)
    exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    parameter.addcdiv_(exp_avg, denominator, value=-learning_rate / bias_correction1)


def warmup_cosine_lr(
    step: int,
    *,
    total_steps: int,
    max_lr: float,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """返回含线性 warmup 的 cosine 学习率；``step`` 范围是 0..total_steps。"""

    if total_steps < 1 or not 0 <= warmup_steps <= total_steps:
        raise ValueError("total_steps 必须为正，warmup_steps 必须位于 0..total_steps")
    if not 0 <= step <= total_steps:
        raise ValueError("step 必须位于 0..total_steps")
    if not 0 <= min_lr <= max_lr:
        raise ValueError("学习率必须满足 0 <= min_lr <= max_lr")
    if warmup_steps and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if warmup_steps == total_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def training_memory_ledger(
    parameter_count: int,
    *,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_state_bytes: int = 8,
    master_weight_bytes: int = 4,
    activation_bytes: int = 0,
    world_size: int = 1,
    shard_parameters: bool = False,
    shard_gradients: bool = False,
    shard_optimizer: bool = False,
) -> dict[str, int]:
    """估算每设备训练内存；结果是账本，不是硬件峰值保证。"""

    integer_values = (
        parameter_count,
        parameter_bytes,
        gradient_bytes,
        optimizer_state_bytes,
        master_weight_bytes,
        activation_bytes,
        world_size,
    )
    if any(not isinstance(value, int) for value in integer_values):
        raise TypeError("内存账本参数必须是整数")
    if parameter_count < 0 or activation_bytes < 0 or world_size < 1:
        raise ValueError("parameter_count/activation_bytes 必须非负，world_size 必须 >= 1")
    if any(value < 0 for value in integer_values[1:-1]):
        raise ValueError("每项字节数不能为负")

    def component(bytes_per_parameter: int, sharded: bool) -> int:
        total = parameter_count * bytes_per_parameter
        return math.ceil(total / world_size) if sharded else total

    ledger = {
        "parameters": component(parameter_bytes, shard_parameters),
        "gradients": component(gradient_bytes, shard_gradients),
        "optimizer_states": component(optimizer_state_bytes, shard_optimizer),
        "master_weights": component(master_weight_bytes, shard_optimizer),
        "activations": activation_bytes,
    }
    ledger["total"] = sum(ledger.values())
    return ledger


@torch.no_grad()
def evaluate_token_logits(
    logits: Tensor,
    labels: Tensor,
    loss_mask: Tensor | None = None,
) -> dict[str, float | int]:
    """计算平均 token loss、perplexity 与 accuracy；不跨 tokenizer 比较 perplexity。"""

    if logits.ndim < 2 or logits.shape[:-1] != labels.shape:
        raise ValueError("logits 应为 [..., vocab]，labels 必须匹配其前置维度")
    if labels.numel() == 0:
        raise ValueError("labels 不能为空")
    if loss_mask is not None and loss_mask.shape != labels.shape:
        raise ValueError("loss_mask 必须与 labels 同形状")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    mask = torch.ones_like(labels, dtype=torch.bool) if loss_mask is None else loss_mask.bool()
    token_count = int(mask.sum().item())
    if token_count == 0:
        raise ValueError("至少需要一个有效评测 token")
    loss = token_loss[mask].mean()
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions[mask] == labels[mask]).float().mean()
    return {
        "loss": float(loss.item()),
        "perplexity": float(torch.exp(loss).item()),
        "accuracy": float(accuracy.item()),
        "token_count": token_count,
    }


def make_next_token_batch(
    tokens: Tensor,
    batch_size: int,
    block_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens 必须是一维")
    if batch_size < 1 or block_size < 1:
        raise ValueError("batch_size 和 block_size 必须为正")
    if len(tokens) <= block_size:
        raise ValueError("token 数必须大于 block_size")
    starts = torch.randint(
        0,
        len(tokens) - block_size,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([tokens[start : start + block_size] for start in starts.tolist()])
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts.tolist()])
    return x, y
