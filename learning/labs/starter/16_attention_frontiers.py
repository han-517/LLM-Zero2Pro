"""Week 29–32：补全滑窗可见性和正特征映射。"""

from torch import Tensor


def sliding_window_mask(query_length: int, key_length: int, window: int) -> Tensor:
    """返回 bottom-right 对齐的 causal window mask，形状 [Tq,Tk]。"""
    # TODO: cache 解码时 query 的绝对位置从 key_length-query_length 开始。
    raise NotImplementedError


def positive_feature(x: Tensor) -> Tensor:
    """线性注意力的正特征映射 phi(x)=elu(x)+1。"""
    # TODO: 不要返回未经约束的 x，否则归一化分母可能失效。
    raise NotImplementedError
