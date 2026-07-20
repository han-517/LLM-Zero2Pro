# MoE 练习

## 1. 总参数与活跃参数

一个 MoE 有 8 个相同 FFN，每 token 选择 2 个。分别写出总专家参数、每 token 活跃专家参数和 Router 参数。说明为什么模型名中的“8×”不能直接理解为 8 倍计算。

## 2. 容量

`tokens=100, experts=8, top_k=2, capacity_factor=1.25`，计算每专家容量。若所有 token 都选择同一个专家，会发生什么？列出三种处理策略。

## 3. 负载与 z-loss

让 Router 权重全为 0，观察 Softmax 概率、Top-k 选择和实际负载。解释为什么“概率完全均匀”仍可能因 tie-breaking 造成选择不均。

## 4. Upcycling

把一个 Dense Expert 复制到 4 个专家。验证复制后各专家输出一致，再给权重加入微小噪声。讨论对称性被打破与能力被破坏之间的权衡。

## 5. 最小路由崩溃

手动让某个 Router logit 极大，记录负载、dropping、balance loss 和 z-loss。分别加入两种辅助损失，说明它们修复的不是同一个问题。

