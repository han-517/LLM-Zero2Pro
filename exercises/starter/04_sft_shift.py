"""Week 40：同步移动 next-token labels 与 assistant mask。"""

from __future__ import annotations

import torch
from torch import Tensor


def causal_sft_targets(token_ids: Tensor, assistant_mask: Tensor) -> tuple[Tensor, Tensor]:
    """输入 [B,T]，返回与 logits[:, :-1] 对齐的 labels 与 loss mask。"""
    # TODO: logits[:, t] 预测 token_ids[:, t+1]。
    raise NotImplementedError


if __name__ == "__main__":
    tokens = torch.tensor([[10, 11, 12, 13, 14]])
    mask = torch.tensor([[0, 0, 0, 1, 1]])
    labels, shifted_mask = causal_sft_targets(tokens, mask)
    assert labels.tolist() == [[11, 12, 13, 14]]
    assert shifted_mask.tolist() == [[0, 0, 1, 1]]
    assert labels.shape == shifted_mask.shape == (1, 4)
    print("通过：SFT labels 与 answer mask 同步 shift")
