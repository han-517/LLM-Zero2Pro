# 第 43 周：DPO 与离线偏好优化的数据假设

## 课程定位

PPO 需要在线采样、奖励模型和 value 系统，工程链很长。Direct Preference Optimization（DPO）从 KL 正则化奖励最大化推导出一个直接比较 policy/reference 序列 log-prob 的离线分类目标。本周实现稳定 DPO loss，重点审计四组 log-prob 是否来自同一模板、mask、聚合与截断。DPO 绕开显式奖励模型训练和在线 PPO，不等于没有 reference、没有偏好假设或没有分布外风险。

## 学习目标

学习者应能从 chosen/rejected 的 policy/reference log-ratio 写出 DPO margin；解释 beta 的尺度和 reference 的锚定作用；用 `-logsigmoid` 稳定实现损失；追踪 sequence log-prob 的 shift、response mask 与 sum/mean 聚合；构造交换 chosen/rejected、policy=reference 和 margin 增大测试；说明离线偏好覆盖、标签噪声、长度偏差与 off-policy 边界。

## 前置知识

需要第 40 周的 response mask、第 42 周 pairwise logistic loss和 KL 直觉。每条数据为 `(prompt, chosen, rejected)`，两个回答共享完全相同的 prompt 与模板。policy 可训练、reference 冻结，四组序列 log-prob 均只聚合回答 token。本周 helper 接收已经聚合的 `[B]` 数值，生成这些数值的模型前向与 mask 契约仍是调用方责任。

## 核心直觉

DPO 不直接问“chosen 的概率高不高”，而问“相对 reference，policy 是否更偏向 chosen 而不是 rejected”。reference 已有的偏好会被抵消；policy 需要改善 chosen/rejected 的相对差。如果 policy 与 reference 完全相同，margin 为 0，模型处在 50% 偏好概率、loss=log2 的起点。beta 把 log-ratio 差缩放为分类 logit；它同时关联隐式 KL 正则强度，但不同代码的符号和参数定义应以公式为准。

## 张量与数据契约

四组 `policy_chosen_logp`、`policy_rejected_logp`、`reference_chosen_logp`、`reference_rejected_logp` 必须同形有限浮点，通常 `[B]`。chosen/rejected 必须使用同一 tokenizer/chat template、prompt 截断和回答边界；policy/reference 必须输入完全相同 token。序列 log-prob 默认为回答 token 的 sum，若改 mean 会改变原始目标和长度偏好，必须写进实验名。reference 参数不求梯度且 revision 固定；adapter 场景下要明确 reference 是关闭 adapter 的同底座还是单独模型。

偏好数据还需记录标注协议、ties/无共识处理、位置随机化、来源、重复、长度分布、安全政策版本与 split。chosen 并非“事实正确”的同义词，它只反映特定采样和标注过程中的相对选择。

## 公式推导与算法机制

定义 policy 偏好差 `Δπ=logπθ(y_w|x)-logπθ(y_l|x)`，reference 差 `Δref=logπref(y_w|x)-logπref(y_l|x)`。DPO 分类 logit

`z=β(Δπ-Δref)`，损失 `L=-mean log σ(z)`。

若 policy 相对 reference 更提高 chosen，z 增大、loss 下降。实现用 `-F.logsigmoid(z)`，避免先算 sigmoid 后 log 在大负数时下溢。论文把这一目标与带 KL 约束的最优策略、Bradley–Terry 偏好模型联系起来；推导依赖偏好数据和奖励建模假设。离线数据不随新策略更新，因此 policy 离数据支持越远，目标越难保证行为质量。

论文常报告隐式 reward `r_chosen=β(logπ_chosen-logπref_chosen)` 与 rejected 对照，它适合诊断 margin，但不是一个独立训练并可任意部署的奖励模型。

## 手算与数值例

设 policy chosen/rejected logp 为 `-2,-4`，reference 为 `-3,-4`，则 `Δπ=2`、`Δref=1`。beta=0.1 时 z=0.1，loss≈0.644；beta=1 时 z=1，loss≈0.313。若 policy=reference，则 z=0、loss≈0.693。若把 chosen/rejected 交换，z 变负、loss 增大。回答很长时 sum logp 绝对值更大，但 DPO 使用差分；长度效应仍不会自动消失，因为 chosen/rejected 长度和每 token 差累积不同。

## 最小代码实现

```python
import math
import torch
from llm_from_scratch.post_training import dpo_loss

pc = torch.tensor([-2.0, -1.0])
pr = torch.tensor([-4.0, -2.0])
rc = torch.tensor([-3.0, -1.5])
rr = torch.tensor([-4.0, -2.5])
loss, chosen_reward, rejected_reward = dpo_loss(pc, pr, rc, rr, beta=0.5)
assert torch.isfinite(loss)

neutral, _, _ = dpo_loss(rc, rr, rc, rr, beta=0.5)
assert abs(float(neutral) - math.log(2.0)) < 1e-6
swapped, _, _ = dpo_loss(pr, pc, rr, rc, beta=0.5)
assert swapped > loss
print(float(loss), chosen_reward.tolist(), rejected_reward.tolist())
```

## 反例、常见误区与调试

反例一 policy 和 reference 使用不同 chat template，log-ratio 差混入模板差异。反例二 chosen 较长且用 sum、rejected 较短，却不报告长度切片。反例三 reference adapter 未真正关闭，四组数值来自同一个活动策略。反例四为了省算力复用过期或错误 tokenizer 的 reference logp。反例五用 `log(sigmoid(z))` 在 z=-1000 时得到 `-inf`。反例六把 DPO 当在线 RL，它不会主动采样当前策略失败，也不会验证 chosen 的事实或安全性。

调试从 policy=reference→log2 开始，再人工增加 policy chosen logp，确认 loss 单调下降；交换对并检查符号；逐 token打印 chosen/rejected mask 和 logp，保证 prompt 未计入；比较 sum/mean 但把它们标为不同目标。训练中报告 preference accuracy、margin、chosen/rejected 长度、KL 代理与基础能力回归，不能只看 loss。

## 主流工作与实现边界

DPO 之后出现 IPO、KTO、ORPO、SimPO 等大量偏好目标，它们改变假设、reference 使用或长度归一化，不能仅因 API 类似便混称 DPO。在线 DPO 变体重新采样策略输出，以缓解静态数据支持问题，但系统复杂度也上升。本周代码只核验给定四组序列 log-prob 的标量目标，不包含模型 forward、reference 缓存、分布式训练、偏好数据去偏或安全评测。

## 实验与 Notebook 对照

运行 `notebooks/core/10_posttraining.ipynb` 的 sequence log-prob 与 DPO 单元；打开 `docs/interactive/training-and-alignment.html` 切到 DPO。补完 `exercises/starter/19_posttraining.py` 的 `sequence_logprob`，执行 `uv run llm-course exercises check 19`。实验扫描 beta 与人为 margin，比较 sum/mean 对长短回答的排序，并构造模板不一致的失败样例。

## 验收标准

合格：policy=reference 时 loss=log2，正 margin 降低 loss，交换 pair 增大 loss。良好：四组 log-prob 的模板、mask、shift、聚合与 reference 冻结均有断言。优秀：按长度、安全和领域切片报告，并能说明 DPO 的 Bradley–Terry、离线覆盖和 KL 假设。把不同模板数值相减、隐式改 mean 或称“不需要 reference 的 DPO”者不通过。

## 一手来源

- DPO 原论文：https://arxiv.org/abs/2305.18290
- DPO 作者官方参考实现：https://github.com/eric-mitchell/direct-preference-optimization
- TRL 官方 DPO Trainer 文档：https://huggingface.co/docs/trl/dpo_trainer
- IPO 对偏好优化泛化的原论文：https://arxiv.org/abs/2310.12036
- KTO 原论文与替代偏好数据设定：https://arxiv.org/abs/2402.01306
