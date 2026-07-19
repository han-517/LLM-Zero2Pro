# 第 42 周：偏好奖励模型与 PPO 系统边界

## 课程定位

SFT 模仿参考答案，却不能直接表达“两个合理回答中人更喜欢哪一个”。RLHF 常先把成对偏好拟合为奖励模型，再用 PPO 在在线采样上优化策略，并以 reference KL 约束漂移。本周实现可手算的 pairwise reward loss 与 clipped policy toy objective，同时逐项列出完整 LLM PPO 还需要的 rollout、value、GAE、mask、KL 和监控，防止把一行 surrogate loss 误称为完整 PPO。

## 学习目标

学习者应能定义 prompt/chosen/rejected 数据契约；从 Bradley–Terry 模型推导 `-log sigmoid(r_c-r_r)`；解释奖励绝对平移不可辨识与 pair accuracy 的局限；从 old/new log-prob 推导 PPO ratio 和裁剪；手算正负 advantage 时裁剪方向；列出完整 RLHF PPO 数据流，并识别 reward hacking、长度偏差、标注偏差和 KL 失控。

## 前置知识

需要 sequence log-prob、logistic loss、策略梯度直觉和 SFT 模板。约定奖励模型输入 chosen/rejected 使用同一 prompt、tokenizer、模板与截断；输出标量 `r:[B]`。PPO 教学 helper 已给定 `new_logp/old_logp/advantages:[...]` 和二值 response mask；old log-prob 与 advantage 都视为固定 rollout 数据，不能让梯度回流到采样策略或奖励计算。

## 核心直觉

奖励模型像把人类二选一压缩为分数差：只要 chosen 比 rejected 高即可，两个分数同加 100 不改变偏好概率。策略随后尝试生成高奖励回答，但奖励模型只是有限标注的代理，过度优化会找到漏洞。PPO 的 ratio 衡量新策略把某动作概率改了多少；裁剪像局部护栏，阻止单个 batch 把概率推得太远，却不能替代 reference KL、可靠奖励和在线安全评估。

## 张量与数据契约

偏好 batch 的 chosen/rejected 必须共享 prompt，回答部分有独立 mask，奖励读取位置（EOS、最后有效 token 或池化）需明确。`chosen_reward`、`rejected_reward:[B]` 为有限浮点同形。PPO token 量常为 `[B,T]`：new/old log-prob、advantages、response mask 完全同形；prompt/padding 不进入 policy loss。rollout 还应保存 prompt、response、旧策略 log-prob、reference log-prob、token value、reward 分解、终止原因与 policy revision。

完整 PPO 中 value/return 的 bootstrap 语义必须区分真正终止与长度截断；KL 是逐 token `logπ-logπ_ref` 的采样估计或其他定义，系数可能自适应。生成温度、max tokens 和 stop 序列改变数据分布，必须随 checkpoint 一同记录。

## 公式推导与算法机制

Bradley–Terry 假设 `P(c≻r)=σ(r_c-r_r)`，最大似然损失

`L_RM=-mean log σ(r_c-r_r)`。

当两分数相等，损失 `log2≈0.693`；差值增大，损失趋零。PPO 定义 `ratio_t=exp(logπ_new-logπ_old)`，最大化 `min(ratio*A, clip(ratio,1-ε,1+ε)*A)`；实现为其负均值。A>0 时不允许 ratio 在增大方向超过 `1+ε` 带来额外收益；A<0 时不允许 ratio 在减小方向越过 `1-ε` 获得过激更新。

完整 LLM PPO 还需：当前策略在线生成；奖励模型与规则打分；减 reference KL；value 网络估计；按终止/mask 计算 returns/GAE；固定 rollout 进行 minibatch 多 epoch 策略与 value 更新；监控 KL、clip fraction、entropy、value error、长度和各奖励分量。教学 helper 只有中间一个标量目标。

## 手算与数值例

奖励 `r_c=2,r_r=1` 时偏好概率 `σ(1)≈0.731`，loss≈0.313；两者同加 10，结果不变。PPO 取 ε=0.2、A=2、old logp=`log0.5`、new logp=`log0.75`，ratio=1.5，未裁剪收益 3，裁剪收益 2.4，取较小 2.4。若 A=-2 且 ratio=0.5，未裁剪 -1、裁剪 -1.6，取较小 -1.6；负号后成为更大惩罚，阻止概率降得过头。

## 最小代码实现

```python
import torch
from llm_from_scratch.post_training import pairwise_reward_loss, toy_ppo_clipped_loss

chosen = torch.tensor([2.0, 0.5])
rejected = torch.tensor([1.0, -0.5])
rm_loss = pairwise_reward_loss(chosen, rejected)
shifted = pairwise_reward_loss(chosen + 100, rejected + 100)
assert torch.allclose(rm_loss, shifted)

old = torch.log(torch.tensor([[0.5, 0.5]]))
new = torch.log(torch.tensor([[0.75, 0.25]]))
adv = torch.tensor([[2.0, -2.0]])
mask = torch.tensor([[1, 1]])
policy_loss = toy_ppo_clipped_loss(new, old, adv, clip_epsilon=0.2, mask=mask)
assert torch.isfinite(policy_loss)
print(float(rm_loss), float(policy_loss))
```

## 反例、常见误区与调试

反例一 chosen/rejected 使用不同截断，奖励模型其实学长度或 EOS 位置。反例二只报告训练 pair accuracy，未按安全、事实、长度切片；高分不等于泛化偏好。反例三 old log-prob 在每个优化 epoch 重新计算，ratio 失去固定基准。反例四 advantage 未 detach，策略 loss 意外更新 value/reward 图。反例五只用 PPO clip 就声称限制了对 reference 的总体漂移；clip 是局部 rollout 约束，仍需独立 KL 监控。反例六把奖励提升等同于真实质量提升，忽略代理漏洞。

调试先用相同分数得到 log2，再交换 chosen/rejected 验证 loss 变大；检查 reward 读取的是最后有效 token而非 PAD。PPO 用 ratio=1、正负 advantage 手算符号；记录 `approx_kl`、clip fraction、entropy、reward/KL/length 分解。若 reward 急升而人工质量下降，应停止优化、找高奖励失败样例并修订数据/评估，而非继续调学习率。

## 主流工作与实现边界

InstructGPT 是 SFT—RM—PPO 的代表流程，但不同系统的 KL、value 初始化、reward whitening 与采样策略并不一致。现代偏好优化也常用 DPO 等离线目标以降低在线 RL 复杂度，但不自动消除数据偏差。课程实现不含模型 rollout、critic、GAE、reference forward、多 epoch buffer、分布式推理训练或安全红队，名称必须保留 `toy_ppo_clipped_loss`。

## 实验与 Notebook 对照

运行 `notebooks/core/10_posttraining.ipynb` 的 reward/PPO 单元，在 `docs/interactive/training-and-alignment.html` 对比 SFT、DPO、GRPO 信号位置。`exercises/starter/19_posttraining.py` 提供 sequence log-prob 基础。实验构造 reward 分数差扫描与 PPO ratio×advantage 网格，画裁剪前后曲线；再收集“高 reward 但明显错误”的人工反例，作为代理边界证据。

## 验收标准

合格：pairwise loss 平移不变、chosen 差值增大时下降，PPO 手算与代码一致。良好：写出完整 rollout 字段与六步 PPO 数据流，正确处理 old/advantage detach 和 response mask。优秀：报告分项 reward、KL、clip、value、长度与人工审计，能区分局部 clip 和 reference KL。把 toy loss 称为完整 PPO 或只用 reward 均值评估者不通过。

## 一手来源

- PPO 原论文：https://arxiv.org/abs/1707.06347
- InstructGPT 的奖励模型与 PPO 流程：https://arxiv.org/abs/2203.02155
- 从人类反馈学习文本摘要的原始工作：https://arxiv.org/abs/2009.01325
- OpenAI 官方 InstructGPT 说明：https://openai.com/index/instruction-following/
- TRL 官方 PPO/Reward Trainer 文档：https://huggingface.co/docs/trl/main/trainer
