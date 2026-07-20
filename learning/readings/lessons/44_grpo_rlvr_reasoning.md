# 第 44 周：GRPO、RLVR 与可验证推理训练

## 课程定位

数学与代码任务可以用答案检查器或单元测试提供相对客观的奖励，推动了 GRPO 与 RLVR（reinforcement learning with verifiable rewards）在推理训练中的应用。本周从同一 prompt 采样一组回答，手算组内标准化 advantage，并接入 token-ratio clipped toy objective；同时明确完整训练还需在线组采样、旧策略、reference/KL、mask、策略刷新、verifier 治理与安全评测。一个 `group_advantages` 函数不是完整 GRPO，一个答案解析器也不是可靠 RLVR 系统。

## 学习目标

学习者应能构造 `[batch,group]` 奖励并沿 group 计算零均值相对优势；处理常数奖励组而不产生 NaN；解释组大小、奖励稀疏和样本相关性；把组优势广播到 `[B,G,T]` 有效回答 token；区分 outcome、process、格式与安全奖励；设计可攻击的 toy verifier 并修复；读懂 DeepSeekMath/DeepSeek-R1 报告与后续 GRPO 变体之间的实现差异。

## 前置知识

需要 PPO ratio/clipping、因果 response mask 和基本抽样。对每个 prompt q，策略以明确的温度、top-p、最大长度采样 G 个 completion；reward `R:[B,G]` 必须由版本化 verifier 产生。`new_logp/old_logp/response_mask:[B,G,T]` 对齐同一 rollout。课程只在给定这些张量后计算 toy objective，不实际训练语言模型。

## 核心直觉

PPO 常用 critic 估计 baseline；GRPO 家族利用同一问题的一组回答互相作参照：高于组均值的回答得到正 advantage，低于均值为负。这样节省独立 value 模型，却把信号质量绑定到“这一组采到了什么”。若一组全错或全对，标准化后优势全零，当前组不提供相对学习信号；增大 G 可能提高遇到正确答案的概率，也线性增加生成与验证成本。

RLVR 的“可验证”只说明存在自动规则，并不说明规则等同于真正任务。数学解析可能只抽取最后一个数字，代码测试可能覆盖不足，格式奖励可能诱导模板投机。策略会优化你写下的 reward，而不是你心中的意图，因此 verifier 需要像安全边界一样做攻击测试、版本化和隔离评估。

## 张量与数据契约

`rewards:[B,G]` 为有限浮点，G≥2；同一行必须来自同一 prompt 与同一采样配置。组优势通常 `A=(R-mean_G R)/(std_G R+eps)`，课程使用总体标准差 `unbiased=False` 并在 std≤eps 时返回全零。token log-prob、old log-prob 和 response mask 为 `[B,G,T]`，mask 二值且至少一个有效 token；奖励广播到 T，但 prompt/padding 不参与 loss。

实验记录至少包括 policy/reference revision、每 prompt 样本数、temperature/top-p、最大长度、stop、旧策略刷新频率、更新 epoch、clip、beta/KL 定义、reward 分解、verifier 版本、有效组比例与实际生成 token。不同论文对长度归一化、KL、裁剪单位和组标准差有变体，名称相同也不能省略公式。

## 公式推导与算法机制

对第 i 个 prompt 的 G 个奖励，`μ_i=1/G Σ_j R_ij`，`σ_i=sqrt(1/G Σ_j(R_ij-μ_i)^2)`，教学优势

`A_ij=(R_ij-μ_i)/max(σ_i,eps)`，若 `σ_i≤eps` 则定义为 0。

因此非退化组的优势均值为 0、方差为 1；对 reward 做正仿射变换不改变优势（忽略 eps）。随后对每个有效 response token 计算 `ratio=exp(logπ_new-logπ_old)`，使用 PPO 类 clipped surrogate。DeepSeekMath 的完整 GRPO 还含采样、旧策略/reference、KL 与多步更新；后续实现对 token/sequence 归一化、KL 和 clipping 有不同改动。教学函数仅核查符号、广播和 mask。

RLVR 常把最终答案正确记 1、错误记 0。若每题单次成功率 p、独立采样 G 次，至少一个正确的概率是 `1-(1-p)^G`；这解释了增加 G 的信号收益，也显示当 p 极低时生成预算会迅速膨胀。

## 手算与数值例

一组奖励 `[0,1,1,0]` 均值 0.5、总体标准差 0.5，优势为 `[-1,1,1,-1]`。常数组 `[1,1,1,1]` 标准差 0，应返回 `[0,0,0,0]` 而非除零。若单次答对率 p=0.1、G=8，至少一个答对概率 `1-0.9^8≈0.57`，但生成成本是单样本八倍且独立假设可能不成立。一个只检查输出是否含字符串“42”的 verifier，会奖励“答案不是 42，但字符串 42 出现了”，这是直接可利用漏洞。

## 最小代码实现

```python
import torch
from llm_from_scratch.post_training import group_relative_advantages, toy_grpo_clipped_loss

rewards = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
advantages = group_relative_advantages(rewards)
assert torch.allclose(advantages[0], torch.tensor([-1.0, 1.0, 1.0, -1.0]))
assert torch.equal(advantages[1], torch.zeros(4))

old = torch.zeros(2, 4, 3)
new = old.clone().requires_grad_()
mask = torch.tensor([[[1, 1, 0]] * 4, [[1, 0, 0]] * 4])
loss = toy_grpo_clipped_loss(new, old, rewards, mask)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(new.grad).all()
print(float(loss.detach()), advantages.tolist())
```

## 反例、常见误区与调试

反例一沿 batch 而非 group 标准化，把不同问题难度混在一起。反例二用默认无偏标准差，数值与论文/实现不一致却不记录。反例三常数组直接除 eps，虽然结果可为零，但极小非零噪声会被放大；显式定义退化组策略。反例四 verifier 只测最终字符串、弱测试或宽松浮点容差，模型学会格式作弊。反例五 rollout 后不断更新 old logp，破坏 ratio 基线。反例六把 pass@k 的采样收益当作模型单样本能力提升。

调试先验证每行 advantage 均值≈0、非退化方差≈1；统计全同奖励组比例；人工维护“reward pass 但语义错”的对抗样例；固定采样 seed 对照 verifier 版本。训练若 reward 升而答案冗长、格式僵化或语言混杂，应检查 reward 分解、长度归一化、KL 和评估泄漏，而非只增加 G。

## 主流工作与实现边界

DeepSeekMath 引入 GRPO 用于数学推理，DeepSeek-R1 报告将 RL、冷启动和多阶段数据结合；DAPO、Dr.GRPO、GSPO 等工作进一步修改采样、归一化或序列级裁剪。它们的公式和训练 recipe 仍在快速演化，不能把所有“组采样+规则奖励”都当成同一 GRPO。课程不包含大规模 rollout engine、异步训练、reference/KL、真实 verifier sandbox、去污染或多轮安全评测。

## 实验与 Notebook 对照

运行 `learning/labs/10_posttraining.ipynb` 的 group advantage 与 toy verifier 单元，打开 `learning/readings/interactive/training-and-alignment.html` 切到 GRPO/RLVR。补完 `learning/labs/starter/19_posttraining.py` 的 `group_advantages` 并运行 `uv run llm-course exercises check 19`。实验扫描 G 与基础成功率 p，测有效组比例；再故意攻击算术答案解析器，提交至少三条 reward hacking 样例及修复测试。

## 验收标准

合格：非退化组优势均值 0、方差 1，常数组精确返回零，toy loss 梯度有限。良好：所有 `[B,G,T]` mask/ratio 契约正确，记录完整采样与 verifier 版本。优秀：报告有效组率、生成 token 成本、KL/长度/安全切片，能解释至少两个 GRPO 变体与原始公式差异，并维护 verifier 对抗集。把组标准化 helper 称为完整 GRPO 或把规则通过率等同真实推理能力者不通过。

## 一手来源

- DeepSeekMath / GRPO 原论文：https://arxiv.org/abs/2402.03300
- DeepSeek-R1 官方论文与代码仓库：https://github.com/deepseek-ai/DeepSeek-R1
- DeepSeek-R1 论文版本：https://arxiv.org/abs/2501.12948
- TRL 官方 GRPO Trainer 实现说明：https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md
- DAPO 开源强化学习系统论文：https://arxiv.org/abs/2503.14476
