# 第 18 周：RoPE 与长上下文扩展——位置进入 Q/K 的相位

## 课程定位

经典 Tiny GPT 把 token embedding 与学习式绝对位置相加。本周把位置移到 attention 内部：对 Q/K 的二维通道对做旋转，使点积自然依赖相对位移。基础 RoPE 是必须从零实现的部分；Position Interpolation、YaRN、LongRoPE、partial RoPE 与 iRoPE 是演化路线，必须分别说明机制和证据，不能统称为“把 `rope_theta` 调大”。

## 学习目标

- 从二维旋转矩阵推导 RoPE 的范数不变与相对位移性质。
- 支持 `[T]` 和 `[B,T]` position IDs、partial rotary dimension 与缓存位置偏移。
- 区分基础 RoPE、位置插值、频率缩放、非均匀缩放和 RoPE/NoPE 层交错。
- 设计训练长度外的外推实验，并识别“可运行”与“能可靠利用”的差别。

## 前置

需要理解正弦位置编码、复数相位、点积、MHA 的 `[B,H,T,Dh]` 形状和 KV Cache。先保证 `Dh` 的旋转部分为正偶数，并能追踪 cached decode 中新 token 的绝对位置 ID。

## 直觉

把相邻两个 head 维度看成平面坐标。位置 `m` 决定旋转角 `mθ_i`，不同维度对使用不同频率。Q 在位置 `m`、K 在位置 `n`，同时旋转后，二者夹角只多出 `(n-m)θ_i`，所以相似度编码相对距离。高频维快速旋转，擅长局部区分；低频维变化慢，承载更长尺度。训练长度之外，相位可能落到模型从未见过的组合，数学上能算不代表模型会用。

## 张量/数据契约

输入 `x:[B,H,T,Dh]`，`positions` 可为共享 `[T]` 或逐样本 `[B,T]`。`rotary_dim<=Dh` 且为正偶数，剩余通道原样通过。cos/sin 在输入 device 上生成，角度计算至少用 FP32，再转换到输入 dtype。基础频率可写为 `theta_i=base^{-i/(rotary_dim/2)}`。训练和 cached decode 必须用同一 base、维度配对方式和 position ID 语义；cache 应保存已经按历史位置旋转的 K。

## 推导与机制

二维旋转矩阵满足 `R_m^T R_n=R_{n-m}`，因此

\[
(R_mq)^\top(R_nk)=q^\top R_{n-m}k,
\qquad \|R_mq\|_2=\|q\|_2.
\]

Position Interpolation 把目标长位置按比例压回训练范围；YaRN 对不同频率区间使用不同缩放并配合 attention temperature；LongRoPE 搜索维度相关的非均匀缩放并分阶段扩展。partial RoPE 只旋转一部分 head 维；iRoPE 在一些层用 RoPE、一些层用 NoPE。它们修改的位置分布、频谱或层布局不同，不能由一个统一旋钮精确代表。

## 数值例

取二维 `q=[1,0]`、`k=[1,0]`，频率 `θ=π/6`。位置 2 的 Q 角度为 `π/3`，位置 5 的 K 角度为 `5π/6`，点积为 `cos(π/2)=0`。把二者位置同时加 7，角差仍是 `3θ=π/2`，点积不变。若只把 K 加 1，点积变为 `cos(4θ)=cos(2π/3)=-0.5`。这是最小的相对位移 oracle。

## 最小代码

```python
def rope_pair(x, positions, base=10_000.0):
    # x: [B,H,T,D], D 为偶数；教学版省略 partial 维和缓存。
    half = x.shape[-1] // 2
    inv = base ** (-torch.arange(half, device=x.device).float() / half)
    angle = positions.to(x.device).float()[:, None] * inv[None, :]
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out
```

课程 `apply_rope` 进一步支持 batched IDs 与 partial RoPE。它仍不是生产实现：生产系统会缓存 cos/sin、融合到 Q/K kernel、处理多种 scaling config 和张量并行。课程没有把 PI、YaRN、LongRoPE 全部伪装成完整实现。

## 反例与调试

最隐蔽的错误是训练时按交错偶奇维配对，推理时按前半/后半配对；范数测试仍会通过，但 checkpoint 语义不兼容。第二个错误是 cached decode 每次给新 token 位置 0，完整前向与缓存结果立刻不一致。第三个错误是旋转 V，基础 RoPE 只作用于 Q/K。设备错误常来自 CPU `positions` 与 CUDA `x` 混用。长上下文实验若只看 loss 不做位置分桶，会掩盖远距离退化。

## 主流工作与证据等级

RoFormer 是基础论文证据；LLaMA、Qwen、DeepSeek 等公开报告证明 RoPE 家族被广泛采用。PI、YaRN、LongRoPE 各有论文实验，但结果依赖继续训练、数据和目标长度。Llama 4 官方说明公开 iRoPE，Qwen3.5 模型卡公开 partial RoPE；这是模型采用证据，不是所有模型都应照搬的结论。2026 新方法应标为 frontier，不能和经过多年复现的基础 RoPE放在同一成熟度等级。

## Notebook、互动图与 starter

在 `learning/readings/interactive/architecture-lab.html` 调节相对位移和旋转比例；在 `learning/labs/06_modern_decoder.ipynb` 画多频率相位并核对同移不变性；完成 `learning/labs/starter/07_rope.py`。互动单位圆只画一个频带，不能替代完整 head 的频谱实验。

## 实验

先做范数、同移点积和完整/缓存 logits 三个 correctness 实验。再用训练长度 128 的复制任务，评测 128、256、512、1024，并按目标距离分桶。比较基础 RoPE、简单 PI 和 partial RoPE；固定训练 token 和参数。结果中注明哪些是自己实现、哪些只是公式模拟，不把没有继续训练的失败归因于方法本身。

## 验收 rubric

- 35%：范数、相对位移、batched IDs、partial 维与 cache oracle 通过。
- 25%：能分别解释 PI、YaRN、LongRoPE、partial/iRoPE。
- 25%：长度外推实验含距离分桶和训练范围标记。
- 15%：明确基础实现、概念模拟与生产融合 kernel 的边界。

## 一手来源

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN](https://arxiv.org/abs/2309.00071)
- [LongRoPE](https://arxiv.org/abs/2402.13753)
- [Llama 4 官方 iRoPE 说明](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
