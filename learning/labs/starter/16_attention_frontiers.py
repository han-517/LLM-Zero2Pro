"""Week 29–32：补全滑窗可见性和正特征映射。"""

from torch import Tensor


def tiled_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size: int = 64,
) -> Tensor:
    """用分块 online softmax 计算精确 causal attention。

    输入采用 ``[batch, heads, time, dim]``；实现不得物化完整的
    ``[time, time]`` 概率矩阵。每个 query block 需要维护运行最大值、
    指数和与 value 累加器，并正确处理最后一个不足 block 的分块。
    """

    # TODO: 校验形状，并按 key/value block 更新 online-softmax 状态。
    raise NotImplementedError


def sliding_window_mask(query_length: int, key_length: int, window: int) -> Tensor:
    """返回 bottom-right 对齐的 causal window mask，形状 [Tq,Tk]。"""
    # TODO: cache 解码时 query 的绝对位置从 key_length-query_length 开始。
    raise NotImplementedError


def positive_feature(x: Tensor) -> Tensor:
    """线性注意力的正特征映射 phi(x)=elu(x)+1。"""
    # TODO: 不要返回未经约束的 x，否则归一化分母可能失效。
    raise NotImplementedError
