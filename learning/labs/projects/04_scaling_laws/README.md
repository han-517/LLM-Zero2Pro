# 大作业 04：受预算约束的 Scaling Law 实验

本项目要求真正训练一组 tiny Transformer，而不是对讲义里预制的三点数组做回归。
你先在 `student_scaling` 预注册运行网格和 FLOP 上限，再复用大作业 01 中自己实现的模型
完成 sweep、保存 checkpoint 与原始 JSONL，最后拟合幂律并检查残差。

## 前置课次

完成第 27、28 课、大作业 01，以及 AdamW/schedule starter 15。CPU 默认网格只用于方法正确性，不用于预测大型 LLM。

## 固定顺序

1. **预算器**：精确计算课程 Decoder 参数量，明确 `6 × parameters × tokens` 只是训练 FLOP 代理。
2. **预注册**：模型尺寸 × token 预算 × seed 形成唯一 run ID；token 向完整 optimizer step 取整后再检查总预算。
3. **真实运行**：每个配置从头初始化并训练，保存配置哈希、checkpoint 哈希、步数、token、初末 loss 与耗时。
4. **前沿**：删除在 compute/loss 上被支配的 run，保留异常点原始记录。
5. **拟合**：在声明的 asymptote 下拟合 `L(C)=E+A C^b`，报告 log-space 残差、R² 与 bootstrap 区间。
6. **边界**：讨论小模型、短训练、优化未收敛、单一数据和 FLOP 代理造成的外推失败。

## 核查与运行

```text
uv run llm-course projects check 04
uv run python learning/labs/projects/04_scaling_laws/run_sweep.py
```

`run_sweep.py` 会从相邻的大作业 01 导入你的 learner-owned `student_lm`，不会导入 `src/` 参考实现。默认网格生成：

```text
artifacts/runs.jsonl
artifacts/checkpoints/*.pt
artifacts/summary.json
```

## 完成标准

- 公开核查通过，预算器和实际模型参数量一致，总预测 FLOPs 不超过预注册上限。
- `runs.jsonl` 至少覆盖 3 个模型尺寸 × 2 个 token 预算；每条记录能追到配置和 checkpoint。
- 报告 Pareto 前沿、幂律系数/指数、R²、逐点残差和 seeded bootstrap 区间。
- 失败、超时和异常 run 不删除；说明是否重跑以及这对统计独立性的影响。
- 不把 tiny CPU 结果包装成大型模型的最优参数/token 配比。