"""Week 41–44：补全 response log-prob 聚合与组内相对优势。"""

from torch import Tensor


def sequence_logprob(token_logprobs: Tensor, response_mask: Tensor) -> Tensor:
    """对 [B,T] 的回答 token 求和；prompt/padding 不计入。"""
    # TODO: 验证 mask 与张量形状，不在这里偷偷做长度归一化。
    raise NotImplementedError


def group_advantages(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """沿组维标准化 [B,G] 奖励；常数组返回零。"""
    # TODO: 使用总体方差，并安全处理零方差。
    raise NotImplementedError
