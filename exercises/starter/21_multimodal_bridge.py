"""选修：补全 patchify 与视觉到文本维度的 projector。"""

from torch import Tensor


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """把 [B,C,H,W] 转为 [B,N,C*P*P]，不改变 patch 内元素顺序。"""
    # TODO: H/W 必须可整除，明确 N=(H/P)*(W/P)。
    raise NotImplementedError


def project_vision(patches: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """把视觉 patch embedding 投到文本 Decoder 的 d_model。"""
    # TODO: 最后一维矩阵乘法，并验证可选 bias。
    raise NotImplementedError
