# 第 21 周：初始化、残差尺度与训练稳定性

## 课程定位

现代组件已经接入，但“代码能前向”不等于“深层训练稳定”。本周建立可观测的稳定性闭环：初始化控制方差，Pre-Norm 保留梯度路径，warmup 控制早期更新，AdamW 解耦权重衰减，gradient clipping 限制异常步。重点是用日志定位根因，不把所有 loss spike 都交给一个超参数。

## 学习目标

- 用 fan-in 假设估算线性层输出方差与残差累积。
- 写出 AdamW 的一阶、二阶矩与 decoupled decay 更新。
- 区分 warmup、残差缩放、QK-Norm、clipping 各自作用点。
- 建立 loss、学习率、梯度、激活与非有限值的诊断顺序。

## 前置

需要掌握随机变量方差、反向传播、Adam、RMSNorm 和残差连接。能够读取优化器 state，并知道 `grad_norm` 必须在 unscale 后计算。分布式场景的全局范数需要 collective，本周先在单进程 reference 上验证。

## 直觉

每个残差 block 都向主干加一份更新；若各分支方差不受控，层数增加会让激活能量漂移。初始化决定第 0 步的信号尺度，归一化决定每步输入尺度，学习率决定参数位移，clipping 只在更新异常大时踩刹车。它们不是互相替代的旋钮。错误数据或 mask 产生的 spike 即使被裁剪，也仍在优化错误目标。

## 张量/数据契约

线性层权重 `[Dout,Din]`，fan-in 为 `Din`。日志至少按 step 保存 loss、LR、全局 grad norm、每层输入/输出 RMS、最大绝对激活、非有限计数和 token/s。AdamW 的 `m,v` 与参数同 shape，常以 FP32 保存；weight decay 通常排除 bias 和归一化 scale，但规则必须显式记录。梯度累积时，optimizer step 与 microstep 不能混淆。

## 推导与机制

若输入独立、零均值、方差 `σ_x²`，权重独立且方差 `σ_w²`，线性输出单维方差近似 `Din σ_w² σ_x²`；取 `σ_w²≈1/Din` 可保持量级。AdamW：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

decay 直接作用参数，不先被 `1/sqrt(v)` 预条件，因此与把 `λθ` 加入 Adam 梯度不等价。warmup 让 `η_t` 从小值上升，降低矩估计尚不稳定时的位移。深层配方还可能缩放 residual projection 或使用 DeepNorm、QK-Norm；这些修改需单独消融。

## 数值例

参数 `θ=2`、学习率 `0.1`、decay `0.01`，AdamW decay 后先得到 `1.998`，再减自适应梯度步。若把 L2 梯度 `0.02` 混入 Adam，它会进入一阶和二阶矩，实际缩放依赖历史 `v`，不再是固定乘 `0.999`。再看 32 个独立残差分支：若每个分支输出方差约 1 且直接相加，主干方差可能随深度增长，说明只靠 fan-in 初始化并不自动解决残差累积。

## 最小代码

```python
import torch


@torch.no_grad()
def adamw_step(p, g, m, v, step, lr, beta1=.9, beta2=.95, wd=.1, eps=1e-8):
    m.mul_(beta1).add_(g, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    p.mul_(1 - lr * wd)  # decoupled decay
    p.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)


p = torch.tensor([2.0])
g = torch.tensor([0.25])
m = torch.zeros_like(p)
v = torch.zeros_like(p)
adamw_step(p, g, m, v, step=1, lr=0.1, wd=0.01)
assert torch.isfinite(p).all() and p.item() < 2.0
```
这是单参数组教学 oracle，未覆盖 foreach/fused kernel、capturable state、分片 optimizer、低精度 master weights 和参数排除规则。正式训练使用框架实现并用单步结果核对。

## 反例与调试

loss 出现 NaN 时先定位第一个非有限张量，不要先把 clipping 从 1 改成 0.1。若 grad norm 正常而激活逐层增大，检查残差尺度和初始化；若只在 warmup 结束时 spike，检查 schedule 连续性；若恢复 checkpoint 后 LR 跳变，scheduler step 和 optimizer step 可能未恢复。混合精度下要先 `unscale_` 再裁剪，否则测到的是放大梯度。Q/K 点积极大时，检查 head scale、RoPE 和 QK-Norm，而非只调 MLP。

## 主流工作与证据等级

AdamW 论文给出解耦正则的基础证据；Pre-LN、DeepNorm 等论文分析深层稳定性。OLMo、LLaMA 等公开报告给出 warmup、clipping、beta 和 decay 配方，是公开采用证据，但训练规模、数据和 batch 不同，超参数不能直接复制。QK-Norm 在多个现代模型中出现，适合作为补充实验，不应写成所有模型的固定要求。

## Notebook、互动图与 starter

在 `notebooks/core/06_modern_decoder.ipynb` 记录深度方向激活/梯度；在 `docs/interactive/training-and-alignment.html` 观察数据和内存变量；使用 `src/llm_from_scratch/training.py` 的优化器/日志工具做单步 oracle。本周无独立 starter 时，交付 notebook 中的故障注入单元和诊断报告。

## 实验

训练 2、8、16 层模型，比较标准正态初始化、fan-in 初始化与 residual projection 缩放；再比较无 warmup、线性 warmup 和故意不连续 schedule。每组注入一次错误 mask 与一次异常大 batch，验证日志能区分数据故障和数值故障。至少报告三个 seed、首个异常 step 与恢复策略。

## 验收 rubric

- 30%：方差和 AdamW 推导正确，单步与框架 oracle 一致。
- 30%：日志能定位第一个异常层/step，而非只展示最终曲线。
- 25%：初始化、warmup、clipping 分别做受控消融。
- 15%：说明教学 optimizer 与 fused/sharded 生产实现边界。

## 一手来源

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
