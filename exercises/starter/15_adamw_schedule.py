"""Week 24：补全 decoupled AdamW 单步与 Warmup-Cosine 学习率。"""

from torch import Tensor


def adamw_step(
    parameter: Tensor,
    gradient: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """返回更新后的 parameter、m、v，不原地修改输入。"""
    # TODO: 偏置校正和 decoupled weight decay 都是验收点。
    raise NotImplementedError


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float) -> float:
    """线性 warmup 后余弦衰减到 0。"""
    # TODO: 明确 step=0、warmup 边界和 total_steps。
    raise NotImplementedError
