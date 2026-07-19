# 多模态选修：把视觉 token 接入文本 LLM

> 边界：本章在 48 周文本主线之外；不下载模型权重，不训练大型视觉语言模型。
> 目标是理解接口、张量形状、数据风险和评测契约，而不是复刻某个商业 VLM。

## 学习目标

完成后应能：

1. 把图像切成 patch，并追踪 \`[B,C,H,W] → [B,N,P²C] → [B,N,D_v]\`。
2. 区分 projector、resampler/cross-attention 和原生交错 token 三类接口。
3. 解释视觉位置、文本位置和 M-RoPE/多轴位置编码要解决什么问题。
4. 识别配对数据噪声、模态长度不平衡、视觉幻觉与文本捷径。
5. 用 CPU toy Notebook 验证维度接口，而不把它误称为完整 VLM。

## 前置知识

- Week 9–15：token、embedding、attention、TinyGPT。
- Week 18：RoPE 只作用于 Q/K。
- Week 20：prefill、KV Cache 与序列长度成本。

## 1. 图像变成 token

给定图像 $x \in \mathbb{R}^{B\times C\times H\times W}$ 和 patch 边长 $P$，
要求 $H,W$ 可被 $P$ 整除。patch 数为：

$$
N = (H/P)(W/P)
$$

每个 patch 展平为 $P^2C$，再经线性层映射到视觉宽度 $D_v$：

$$
X_v = \operatorname{reshape}(x)W_{\text{patch}} + b
$$

这与文本 tokenizer 不同：patchify 是固定几何切分；视觉 embedding 和位置参数
才是可学习部分。ViT 展示了“patch 序列 + Transformer”的基本范式，CLIP
展示了图文对比学习如何对齐两个编码器的表示空间。

## 2. 三类接入接口

### Projector + token prefix

$$
Z = X_vW_p,\qquad W_p \in \mathbb{R}^{D_v\times D_{\text{text}}}
$$

投影后的视觉 token 直接放到文本 token 前或占据特殊占位符位置。接口最小，
但视觉 token 多时会显著增加 Decoder prefill 和 KV Cache。

### Resampler / cross-attention

一组较少的可学习 query 通过 cross-attention 从视觉 token 读取信息，把
$N$ 个视觉 token 压成 $M\ll N$ 个 latent。文本 Decoder 可以消费这些 latent，
或在若干层插入 cross-attention。压缩提高效率，但可能损失小目标、文字和空间关系。

### 原生交错 token

图像、视频帧和文本 token 在一个统一序列中交错。接口更统一，但必须明确：

- 哪些 token 能互相注意；
- 视觉与文本位置如何编码；
- 不同模态的词表/embedding 是否共享；
- 长图像序列是否淹没文本损失；
- 多图和视频的边界 token 如何定义。

## 3. 位置：2D、时间与 M-RoPE

文本 RoPE 的 position ID 通常是一维。图像至少有行、列两轴，视频再增加时间
轴。多轴旋转位置编码把 head 维度划分给不同坐标轴，使 Q/K 点积同时带有多种
相对位移信息。

必须避免的误解：

- M-RoPE 仍然是 attention 内部的 Q/K 变换，不是独立网络层。
- 视觉网格位置不等于真实世界几何；裁剪、缩放和拼接会改变坐标语义。
- 增加最大视觉 token 数不等于模型真正利用了高分辨率细节。

## 4. 数据契约

一个可审计的图文样本至少记录：

- 图像来源、许可、哈希与变换历史；
- 文本来源，是人工描述、OCR、网页邻近文本还是模型合成；
- 图像尺寸、patch/resize 策略、语言和内容类别；
- 训练/验证/测试去重与近重复策略；
- 人脸、隐私、版权、成人内容和地域偏差处理；
- 是否存在“只读文本就能答”的语言捷径。

批次中视觉 token 往往远多于文本 token。若直接按 token 平均 loss，模态贡献
可能失衡；若只监督文本回答，模型也可能学会忽略视觉细节。需要报告采样比例、
每模态 token 数和 loss 权重。

## 5. 视觉幻觉与评测

只看答案流畅度不够。至少拆分：

- object existence：声称的物体是否真的出现；
- attribute：颜色、数量、材质是否正确；
- relation：左右、遮挡、包含关系是否正确；
- OCR/document：文字与布局是否忠实；
- grounding：回答能否对应图像区域或证据；
- text-only control：去掉图像后性能是否几乎不变。

评测集也会污染；网页图文和公开 benchmark 可能进入预训练语料。报告中必须把
事实、推断和限制分开。

## 6. CPU 实验与验收

打开：

- \`notebooks/optional/80_multimodal_bridge.ipynb\`
- \`docs/interactive/multimodal-flow.html\`
- \`exercises/starter/21_multimodal_bridge.py\`

实验只完成：

$$
\text{patchify}\rightarrow\text{vision embedding}\rightarrow
\text{projector}\rightarrow\text{text Decoder interface}
$$

验收：

- patch 数和展平顺序正确；
- H/W 不可整除时显式报错；
- projector 输出最后一维等于文本 \`d_model\`；
- 能比较三种接口的 token 数、prefill 成本和信息瓶颈；
- 能明确说明 toy 实现没有训练视觉编码器、没有多模态数据、没有完整 VLM loss。

## 一手来源

- [ViT](https://arxiv.org/abs/2010.11929)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Flamingo：resampler + cross-attention](https://arxiv.org/abs/2204.14198)
- [BLIP-2：Q-Former](https://arxiv.org/abs/2301.12597)
- [LLaVA：视觉 projector 教学参照](https://arxiv.org/abs/2304.08485)
- [Qwen2-VL：M-RoPE 与动态分辨率](https://arxiv.org/abs/2409.12191)
