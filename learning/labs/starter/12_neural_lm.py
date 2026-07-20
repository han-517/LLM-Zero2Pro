"""Week 7–8：补全 Bigram 计数与显式循环状态更新。"""

from torch import Tensor


def bigram_counts(tokens: Tensor, vocab_size: int) -> Tensor:
    """从一维 token 流返回 [V,V] 次数矩阵。"""
    # TODO: 检查输入，再统计每个相邻 token 对；不要跨文档拼接。
    raise NotImplementedError


def rnn_step(x: Tensor, h: Tensor, w_xh: Tensor, w_hh: Tensor, bias: Tensor) -> Tensor:
    """计算 tanh(x @ W_xh + h @ W_hh + b)。"""
    # TODO: 明确 [B,D]、[B,H] 和权重形状。
    raise NotImplementedError
