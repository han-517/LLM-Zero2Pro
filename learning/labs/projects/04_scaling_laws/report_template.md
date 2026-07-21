# 大作业 04 Scaling Law 报告

## 预注册

- 固定 FLOP 预算：
- 模型尺寸、token 预算与 seed 网格：
- 主要指标、异常 run 判定和停止规则：
- 6ND 代理与真实硬件 FLOPs 的差异：

## 原始运行

附上 `artifacts/runs.jsonl`。不得只粘贴整理后的三点数组；每条记录必须能追到配置、
checkpoint 哈希、步数、token 数、耗时、初末 loss 和随机种子。

## 拟合与残差

- 幂律形式、asymptote 的选择依据：
- exponent 与 bootstrap 置信区间：
- R²、逐 run 残差和 Pareto/等 FLOPs 前沿：
- 对异常 run 的处理（保留原始记录）：

## 外推边界

说明 tiny CPU sweep、优化未收敛、离散超参和单一数据分布为何不能直接预测大型 LLM。
