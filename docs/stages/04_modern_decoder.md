# 阶段四：现代 Decoder 组件——从经典 GPT 到 RoPE + GQA

> 本章对应第 16–21 周。目标不是背组件名，而是把 RMSNorm、SwiGLU、RoPE、
> GQA 与 KV Cache 真正接进同一个可训练 Decoder，并用经典配置作受控对照。

## 1. 两个可运行的配置预设

```python
classic = GPTConfig.classic(vocab_size=32000)
modern = GPTConfig.modern(vocab_size=32000, n_head=16, n_kv_head=4)
```

| 组件 | `classic` | `modern` |
|---|---|---|
| 归一化 | LayerNorm | RMSNorm |
| MLP | GELU 两层 MLP | SwiGLU |
| 位置 | 学习式绝对位置 | RoPE |
| KV heads | 等于 Q heads | 默认 GQA，可设 1 得 MQA |

两者共享数据、训练与生成 API。预设是教学对照，不声称复刻某个具体开源模型。

## 2. RMSNorm：不做均值中心化

\[
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot w
\]

LayerNorm 会减均值并按方差缩放；RMSNorm 不减均值，因此不强制输出均值为零。
“保留均值信息”容易被误解为均值完全不变，应避免这种说法。参考实现用 FP32
计算平方、归约和倒平方根，再转回输入 dtype。

测试：零输入有限；公式一致；不同 dtype 下无 NaN；梯度可回传。

## 3. SwiGLU：门控的前馈层

\[
\operatorname{SwiGLU}(x)=
(\operatorname{SiLU}(xW_g)\odot xW_u)W_d
\]

SwiGLU 有两个升维投影。若普通 MLP 隐藏宽度是 `4D`，直接把 SwiGLU 也设成
`4D` 会增加大量参数；课程默认用接近 `8D/3` 且按 8 对齐的宽度作公平近似。

## 4. RoPE：位置进入 Q/K 的旋转

对第 `i` 个二维通道对，用频率 `theta_i` 旋转位置 `m` 的向量：

\[
R_{m,i}=\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}
\]

关键性质：

\[
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
\]

因此点积依赖相对位移。`apply_rope` 支持 `[T]` 与 `[B,T]` position IDs，并把
IDs 移到输入设备；`rotary_dim` 允许只旋转 head 的前一部分，剩余维度是 NoPE。

在注意力层中必须先旋转当前 Q/K，再把已经旋转的 K 写入 cache。若先缓存未旋转
K，后续 decode 无法仅靠一个当前 position 正确恢复历史位置。

### 长上下文扩展不是一个旋钮

- Position Interpolation：把更长位置压回训练位置范围。
- YaRN：结合频率相关缩放与温度修正。
- LongRoPE：搜索非均匀维度缩放并进行渐进扩展。
- partial RoPE：只旋转一部分维度。
- iRoPE：在 RoPE 与 NoPE 层之间交错。

课程代码完整实现基础/partial RoPE；其余方法在图和讲义中讲原理，不应把“修改
`rope_theta`”写成这些方法的等价实现。

## 5. MHA、MQA 与 GQA

设 Query heads 为 `Hq`、KV heads 为 `Hkv`：

- MHA：`Hkv = Hq`；
- MQA：`Hkv = 1`；
- GQA：`1 < Hkv < Hq`，且 `Hq` 能被 `Hkv` 整除。

课程 GQA 按 KV head 分组计算，不用 `repeat_interleave` 物化重复 K/V。MQA 是同一
实现的边界情况，必须单独测试。

每层 K/V cache 字节数近似为：

\[
B\times T\times 2\times H_{kv}\times D_h\times\text{bytes-per-element}
\]

全模型还要乘层数 `L`。减少 KV heads 直接降低缓存和读取带宽，但 Q heads 不变。

## 6. Prefill、Decode 与组合 mask

Prefill 并行处理 prompt；decode 每步只产生新 Q/K/V，并让新 Q 查询历史 cache。
因果 mask 必须 bottom-right 对齐，padding mask 再与其组合。完全遮蔽的 Query 行
必须输出零，不能产生均匀权重或 NaN。

`TinyGPT.forward()` 支持：

- 显式 `position_ids=[T]` 或 `[B,T]`；
- bool/浮点 `attention_mask`；
- `return_caches=False`，训练时不返回推理 cache；
- 经典绝对位置与现代 RoPE/GQA 共用同一接口。

## 7. 初始化与优化稳定性

初始化要观察残差深度方向的激活 RMS 与梯度，而不只看第一步 loss。AdamW 把权重
衰减与自适应梯度更新解耦，通常不能简单等同于“向 Adam 梯度加入 L2”。warmup
降低训练初期更新冲击；gradient clipping 只能限制异常更新，不能修复错误数据、
mask 或目标。归一化、router 和大归约在低精度下常需 FP32。

建议记录 loss、学习率、全局梯度范数、各层激活 RMS、token/s 与非有限值计数。

## 8. 实验、边界与验收

实验矩阵固定数据、seed、参数量区间与训练 token：

1. classic vs modern 的单 batch 过拟合；
2. MHA/MQA/GQA 的 cache bytes；
3. RoPE 同移位置后的 logits 不变性；
4. full forward 与逐 token cached logits 一致；
5. cache 满时重建滑动窗口，并区分绝对位置重新编号与相对 RoPE。

教学实现强调公式和 correctness，不包含 fused RMSNorm/SwiGLU、FlashAttention、
张量并行或生产级长上下文缩放。速度结论必须在真实 kernel 和目标硬件上复测。

## 常见误区

- RMSNorm 不减均值，不代表输出均值等于输入均值。
- KV Cache 减少重复计算，却增加持久内存。
- FlashAttention 优化临时 IO，不会自动减少持久 KV Cache。
- RoPE 支持更长 position ID，不等于模型能可靠利用更长上下文。
- GQA 的缓存节省必须包含 batch、层数、dtype，不能只写 `2*T*Hkv*Dh`。

## 一手来源

- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN](https://arxiv.org/abs/2309.00071)
- [LongRoPE](https://arxiv.org/abs/2402.13753)

