# 第 31 周：MLA 与低维 KV 表示

## 课程定位

MQA/GQA 通过减少 KV heads 压缩缓存，Multi-head Latent Attention（MLA）把历史 K/V 信息压进低维 latent。本周必须把三件事分开：latent storage compression、每步重建全部历史 K/V 的教学 baseline、以及通过矩阵吸收和 decoupled RoPE 改写的生产 decode。只实现第一、二件不能声称复现完整 MLA 加速。

## 学习目标

- 推导 dense KV 与 latent cache 的字节比。
- 解释为什么位置相关 RoPE 分量妨碍静态投影吸收。
- 实现并诚实名名的 latent-cache reconstruction baseline。
- 分项报告存储、历史重建 MAC 与 absorbed 路径边界。

## 前置

需要掌握低秩投影、矩阵结合律、RoPE、MHA/GQA 和 cached decode。应能区分参数压缩、激活压缩、cache 压缩与计算压缩，它们不是同一指标。

## 直觉

完整 K/V 像为每个 head 保存两本历史笔记；MLA 先把共同信息压成一本较薄的 latent 笔记，需要时通过不同投影读取。朴素教学版每次把整本历史重新展开成所有 K/V，虽然书架省空间，却可能增加读取计算。生产 MLA 通过把可线性组合的投影吸收到 Query/输出路径，避免显式重建；RoPE 的位置旋转随 token 变化，因而常把一小部分 positional key 单独保留。

## 张量/数据契约

输入 `x:[B,T,D]`，latent `c=W_down x:[B,T,Dc]`，cache 保存 `[B,Tpast,Dc]`。教学重建 `K=W_kup c`、`V=W_vup c` 后 split 为 heads。成本函数输入 `B,L,T,D,Dc,dtype_bytes`，输出 dense K/V bytes、latent bytes 与每 decode step 对全部历史做两次 up projection 的 MAC。完整 MLA 还需 query compression、decoupled RoPE dimensions 和 absorbed weight layout，不能从 baseline shape 推断。

## 推导与机制

dense 每层缓存元素约 `2BTD`（把所有 heads 合并回 D），latent 为 `BTDc`，理论存储比为 `2D/Dc`。朴素重建每步对历史长度 T 做 K、V 两次 `Dc→D` 投影，约 `2BTDcD` MAC/层。若内容 key `K_c=CW_k`，分数 `QK_c^T=QW_k^T C^T`，可把 `W_k^T` 吸收到 Q；value/output 也可结合。但 `R_position(CW_kr)` 中旋转依赖位置，不能作为固定矩阵简单吸收，因此 decoupled RoPE 保留单独位置通道。

## 数值例

`D=4096,Dc=512,B=1,L=32,T=8192,FP16`。dense K/V 约 `1×32×8192×2×4096×2≈4 GiB`；latent 约 `1×32×8192×512×2≈256 MiB`，理论 16 倍压缩。可是 baseline 单步历史重建约 `32×8192×2×512×4096≈1.1e12 MAC`，清楚展示“存储更小”不自动等于“decode 计算更小”。

## 最小代码

```python
import math

import torch
from torch import nn


def split_heads(x, num_heads):
    batch, time, width = x.shape
    if width % num_heads:
        raise ValueError("width 必须能被 num_heads 整除")
    return x.view(batch, time, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, time, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, time, heads * head_dim)


class LatentCacheMLABaseline(nn.Module):
    def __init__(self, d_model=8, latent_dim=4, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, d_model, bias=False)
        self.v_up = nn.Linear(latent_dim, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, latent_cache=None):
        q = split_heads(self.q_proj(x), self.num_heads)
        new_latent = self.kv_down(x)
        latent = (
            new_latent
            if latent_cache is None
            else torch.cat((latent_cache, new_latent), dim=1)
        )
        k = split_heads(self.k_up(latent), self.num_heads)
        v = split_heads(self.v_up(latent), self.num_heads)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        query_pos = torch.arange(q.shape[-2]) + k.shape[-2] - q.shape[-2]
        key_pos = torch.arange(k.shape[-2])
        causal = key_pos[None, :] <= query_pos[:, None]
        weights = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
        y = weights @ v
        return self.output(merge_heads(y)), latent


torch.manual_seed(0)
model = LatentCacheMLABaseline()
x = torch.randn(1, 3, 8)
full, _ = model(x)
_, prefix_cache = model(x[:, :2])
cached_last, _ = model(x[:, 2:], prefix_cache)
torch.testing.assert_close(full[:, -1:], cached_last)
```
类名刻意包含 `Baseline`。它用于 shape、cache bytes 与 full/cached correctness，不包含 DeepSeek 的 query compression、矩阵吸收、decoupled RoPE 或融合 kernel。

## 反例与调试

把 baseline 命名为 `MLA` 后只展示 cache 压缩比，会隐去每步历史重建。成本表必须同时列 bytes 与 MAC。cached/full 不一致时检查 latent 拼接顺序和 Query 对齐；RoPE 若直接施加到完全压缩 key，再尝试静态吸收，会丢位置依赖。低 `Dc` 输出 shape 仍正确但信息瓶颈可能严重，必须测质量。生产论文的速度数字不能拿 Python baseline 复现。

## 主流工作与证据等级

DeepSeek-V2 提出并公开 MLA 架构，是基础模型报告/论文证据；DeepSeek-V3 延续 MLA 并提供更大规模采用证据；V3.2 的 DSA 建在 MLA 上。其他模型可能采用 GQA、MQA 或混合状态，说明 MLA 是重要路线而非统一标准。对 projection absorption 的描述应以论文公式和官方实现为准，第三方简化图只能作辅助。

## Notebook、互动图与 starter

在 `learning/readings/interactive/architecture-lab.html` 切换 `MLA baseline`，观察边界提示；在 `learning/labs/08_attention_frontiers.ipynb` 比较 dense/GQA/latent bytes 与重建 MAC；完成 starter `17` 的 latent-cache 部分。互动图只画一条 latent，不表示完整吸收路径。

## 实验

实现 `Dc={D/2,D/4,D/8}` baseline，核对 full/cached 输出、cache shape 和实际字节。用 profiler 或公式记录每步重建成本，比较 GQA。再手推一个不含 RoPE 的内容 key 吸收等式并用小矩阵验证；加入位置旋转后展示为何固定吸收失败。报告不得把吸收推导的 MAC当作已实现墙钟。

## 验收 rubric

- 30%：latent cache 与 full/cached correctness 通过。
- 25%：bytes、重建 MAC 与压缩比计算完整。
- 25%：能推导内容投影吸收并解释 decoupled RoPE。
- 20%：baseline、完整 MLA 与生产 kernel 边界措辞准确。

## 一手来源

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [DeepSeek-V2 官方代码](https://github.com/deepseek-ai/DeepSeek-V2)
