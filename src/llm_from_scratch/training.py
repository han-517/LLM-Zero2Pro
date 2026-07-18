from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int = 20260718) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_next_token_batch(
    tokens: Tensor,
    batch_size: int,
    block_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens 必须是一维")
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

