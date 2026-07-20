# 第 27 周：Scaling Laws 与计算最优训练

## 课程定位

本周把前几周的数据、优化器、混合精度与分布式资源账本转换成可证伪的规模预测。Scaling law 不是自然常数，而是在指定数据分布、tokenizer、架构、优化配方和训练充分程度下观察到的经验规律。课程只使用缩尺实验学习拟合、残差、IsoFLOP 与计算最优决策；不会把几个 Tiny GPT 点外推成商业模型的确定预算。可靠评测、去污染和故障诊断独立放在第 28 周，避免把“预测训练 loss”与“测量任务能力”混成同一主题。

## 学习目标

完成后应能拟合参数量 $N$、训练 token $D$、计算量 $C$ 与 validation loss 的幂律；解释 Kaplan 与 Chinchilla 结论为何不能机械合并；在固定计算预算下构造 IsoFLOP 曲线并找近似最优 $N,D$；报告随机种子、失败 run、残差、置信区间和观测范围；区分经验拟合、工程估算与生产训练决策；说明 dense Transformer 的 $6ND$ 只是量级近似，对长序列、MoE、状态空间模型和低利用率实现都需重新记账。

## 前置

需要理解对数、线性/非线性回归、训练与验证 loss、参数量、token 数、FLOPs、置信区间，以及第 25 周资源账本和第 26 周并行效率。每个 run 必须已通过单 batch 过拟合和基本稳定性检查；未收敛、loss spike 或数据错误不能悄悄当成正常 scaling 点。

## 直觉

把多个训练 run 想成在有限区域测量一座山坡。幂律拟合描述该区域的平均坡度，残差告诉我们地形是否偏离模型，观测区外则可能出现新的地貌。计算最优不是“模型越大越好”或“固定每参数若干 token”，而是在一次训练计算预算内平衡模型容量与数据覆盖。部署时如果推理量巨大，训练得更久、模型更小可能有更低总成本，因此 Chinchilla 风格的训练最优也不等同生命周期最优。

## 张量/数据契约

每条 run 记录非 embedding 参数 `N`、训练 token `D`、估算 FLOPs、实测设备时、数据与 tokenizer 快照、架构 hash、global batch、optimizer/scheduler、最大学习率、warmup、最终及中间 validation loss、seed、硬件、软件版本和失败状态。比较模型时 validation 文本和 token 边界必须固定。表格至少包含三个规模、每个规模多个 seed；如果删除异常点，要保存原因和包含/排除两版拟合。

MoE 需要同时记录总参数和每 token 激活参数；长序列模型需要显式加入 attention 或替代 mixer 的成本；并行训练还要区分理论模型 FLOPs 与实际 wall-clock。`C` 的定义在所有 run 中必须一致，否则同一坐标轴没有意义。

## 推导与机制

一维形式常写为 $L(x)=E+Ax^{-\alpha}$，其中 $E$ 是不可约项。联合模型可写

$$L(N,D)=E+AN^{-\alpha}+BD^{-\beta}.$$

对 dense Transformer，训练计算常粗略写成 $C\approx6ND$。固定若干预算 $C_i$，选择多组 $N,D$ 且近似满足预算，训练后在每条 IsoFLOP 曲线上寻找最低 loss；再拟合 $N_{opt}(C)$ 与 $D_{opt}(C)$。直接对 $\log L$ 做直线拟合等价于假设 $E=0$，往往不成立；更完整的研究应进行带约束非线性拟合、bootstrap，并检查不同初始化和数据顺序的相关误差。

训练不足会让小模型或大模型系统偏离曲线。学习率、batch、warmup 和正则化若未随规模合理调优，得到的是“当前配方的 scaling”，不是架构上限。计算最优结论还依赖目标函数；若关心下游任务、推理成本、能耗或数据获取成本，应扩展目标而非只最小化预训练 loss。

## 数值例

`N=1e8,D=2e9` 时粗略训练计算为 $1.2e18$ FLOPs。若相同预算把参数翻倍到 `2e8`，token 需近似减半到 `1e9`；两次 run 的 validation loss 才构成一个预算上的容量—数据对照。设观测点的 log-log 斜率为 -0.08，但最大两个模型残差都为正，这提示大模型未充分训练、数据瓶颈或简单幂律失配，不能只报告漂亮的 $R^2$。

## 最小代码

```python
import torch

def fit_power_law(scale, reducible_loss):
    x = torch.log(torch.as_tensor(scale, dtype=torch.float64))
    y = torch.log(torch.as_tensor(reducible_loss, dtype=torch.float64))
    if (torch.as_tensor(reducible_loss) <= 0).any():
        raise ValueError("loss - E must be positive")
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coeff = torch.linalg.lstsq(design, y).solution
    prediction = design @ coeff
    return {"log_A": coeff[0], "slope": coeff[1], "residual": y - prediction}

def dense_training_flops(non_embedding_params, tokens):
    return 6.0 * non_embedding_params * tokens
```

这是教学 baseline：调用者必须先给定或估计 $E$，并且没有处理 seed 层级、失败 run、误差相关性与超参数未调优。生产研究应保存原始表格并使用可审计的统计脚本；拟合函数输出不是训练预算的自动审批器。

## 反例与调试

只用三个点、每点一个 seed 会把噪声当斜率；挑每条曲线最低的一次而不报告所有 run 会产生选择偏差。用训练 loss 比较不同 tokenizer 会改变目标单位；把 embedding 参数一会儿计入、一会儿排除会破坏横轴。小模型使用充分调优的学习率而大模型沿用不稳定配方，会把优化失败解释成规模收益饱和。把 $6ND$ 当实测耗时忽略 attention、重计算、通信和 MFU，也会给出虚假资源结论。

## 主流工作与证据等级

Kaplan 等人的工作建立了语言模型规模经验关系，Chinchilla 在其数据、模型和预算范围内重新研究计算最优分配。DeepSeek LLM 等公开报告展示了重新拟合超参数与 IsoFLOP 的必要性，Beyond Chinchilla-Optimal 则把推理需求纳入训练决策。论文和开放 artifacts 属较强证据；机构模型报告是重要但可能难以独立复现的证据；博客可帮助理解，不能承载精确系数。所有系数必须附数据、日期、架构与拟合范围。

## Notebook、互动图与 starter

在 `learning/labs/07_pretraining_systems.ipynb` 中交互调整 $E,\alpha,\beta$、观测噪声和失败 run，观察拟合线与残差如何变化。互动图应同时显示 log-log 主图、残差图和观测范围阴影，超出范围的预测用虚线。starter 留出 metadata 校验、IsoFLOP 分组、非线性拟合和 bootstrap 空缺；核查器验证单位一致、失败点保留、固定 seed 复现和外推标记。

## 实验

训练至少三个模型规模、每个三 seed，并在至少两个 token 预算记录中间与最终 loss。先固定数据、tokenizer、batch token、scheduler 比例和验证集，再画 IsoFLOP、联合拟合与残差。做一次反事实实验：故意让最大模型学习率过高，观察残差如何暴露配方错误。外推不超过最大观测计算的四倍，并同时给出“不外推”的决策方案。

## 验收 rubric

及格要求是 metadata、单位、拟合与残差完整，能手算 $6ND$ 并说明近似边界。良好要求是包含多个 seed、IsoFLOP 最优点、bootstrap 区间和失败 run 敏感性分析。优秀要求是比较至少两种拟合模型，把通信或推理成本纳入决策，明确区分原论文结论、官方报告和本地缩尺推断，不将观测范围外的系数写成通用规律。

## 一手来源

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [DeepSeek LLM: Scaling Open-Source Language Models](https://arxiv.org/abs/2401.02954)
- [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448)
