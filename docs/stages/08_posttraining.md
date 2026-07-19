# 阶段八：后训练——从“续写文本”到“按要求回答”

## 逐周讲义导航

> 本页是阶段知识地图，用于预习和复盘；完整推导、代码、反例、实验与验收请进入下面的逐周讲义。

- [第 40 周：指令数据与 SFT](../weeks/40_sft_data_contract.md)
- [第 41 周：LoRA 与 QLoRA](../weeks/41_lora_qlora.md)
- [第 42 周：偏好、奖励模型与 PPO](../weeks/42_reward_model_ppo.md)
- [第 43 周：DPO：绕开奖励模型的偏好优化](../weeks/43_dpo_preference_optimization.md)
- [第 44 周：GRPO、RLVR 与推理训练](../weeks/44_grpo_rlvr_reasoning.md)

后训练不是一个损失函数，而是一条数据、监督、偏好、在线采样、验证器和安全评测共同组成的
流水线。本章把“教学最小实现”和“生产训练系统”分开命名，避免跑通一个公式就误以为复现了算法。

## 六周学习顺序

1. 对话模板、response-only SFT 与序列 log-prob。
2. LoRA、merge/unmerge 与 QLoRA 数据流。
3. 偏好数据、奖励模型和 RewardBench。
4. PPO 与 DPO：在线和离线偏好优化。
5. GRPO、RLVR 和可验证奖励。
6. 安全评测、红队和训练前后对照。

推荐先读 [InstructGPT](https://arxiv.org/abs/2203.02155) 的完整流水线，再逐段运行
`src/llm_from_scratch/post_training.py` 和 `tests/test_post_training.py`。训练算法的名字应对应它实际
包含的组件，而不是只对应其中一行公式。

## 1. SFT：模板、shift 和 loss mask 是同一份契约

监督微调仍是 next-token prediction，只是样本被序列化为角色化对话。常见的 response-only
契约只在 assistant token 上计算损失：

```text
tokens:         [BOS, system, user, assistant, answer ..., EOS, PAD]
assistant_mask: [0,   0,      0,    0,         1 ...  1,   0]
labels:         [-100 ...                       answer ..., EOS, -100]
```

response-only 是常见选择，不是普适定律。有些系统会训练 system/user token，有些多轮数据只训练
最后一轮回答。课程要求每个数据集明确记录以下契约：

- 使用的 chat template、BOS/EOS 和 role token；
- assistant 起始标记是否参与损失，EOS 是否参与损失；
- 多轮对话训练全部 assistant 回合还是最后一回合；
- padding side、截断方向、packing 后是否隔离跨样本边界；
- tool call、tool result 和不可见 reasoning token 如何掩码。

`response_only_collator` 将 token、assistant mask 和 padding 变成 batch，并把非回答标签设为
`-100`。`causal_sft_loss` 再同步执行 next-token shift：`logits[:, t]` 预测
`tokens[:, t+1]`。不能只移动 logits 或只移动 mask。

`sequence_logprob` 对同一 shift 后的 token log-prob 做 `sum` 或 `mean` 聚合。偏好学习通常使用
序列总和；均值会改变长度偏好，必须作为显式实验变量记录。测试至少覆盖：

- 改变 prompt 位置的 logits 不改变 response-only loss；
- 改变第一个 answer token 对应的前一位置 logits 会改变 loss；
- padding、空 mask、非法二值 mask 和非有限 logits 被明确处理。

## 2. LoRA 与 QLoRA

[LoRA](https://arxiv.org/abs/2106.09685) 冻结原权重，只学习低秩修正：

```text
W' = W + (alpha / rank) * B A
A: [rank, in_features]
B: [out_features, rank]
```

rank、alpha、dropout 和 target modules 都是训练配置的一部分。只报“用了 LoRA”无法复现；应列出
attention 的 q/k/v/o、MLP 的 gate/up/down 中究竟适配了哪些层。LoRA 减少可训练参数，但激活、
优化器外的临时张量和底座前向仍占显存。

教学 `LoRALinear` 支持 `merge_` 和 `unmerge_`。merge 把低秩增量写入底座，便于无额外分支的
推理；unmerge 恢复训练结构。带 dropout 时只能在 eval 模式合并，并需要测试：

1. eval 下 merge 前后输出相同；
2. unmerge 后权重和输出恢复；
3. 重复 merge/unmerge 不会累计写入；
4. 底座参数保持冻结，只有 A/B 得到梯度。

[QLoRA](https://arxiv.org/abs/2305.14314) 不是“把 LoRA 也变成 4 bit”。论文组合了冻结底座的
4-bit NormalFloat（NF4）、double quantization、paged optimizers 和较高精度计算。量化存储 dtype、
反量化计算 dtype、梯度 dtype 与优化器状态 dtype 要分开报告。课程不声称最小 LoRA 类复现了
QLoRA 内核。

## 3. 偏好数据与奖励模型

成对偏好样本包含 prompt、chosen、rejected 以及标注协议。Bradley–Terry/logistic 形式常写为：

```text
L_RM = -log sigmoid(r_chosen - r_rejected)
```

奖励标量通常来自序列末端或 EOS hidden state；padding 与 EOS 选择错误会让模型读到错误位置。
成对数据还要记录 ties、标注者分歧、位置偏差、长度偏差和安全政策版本。奖励模型只是标注协议的
代理，过度优化会导致 reward hacking。

评测不能只看训练集 pair accuracy。[RewardBench](https://arxiv.org/abs/2403.13787) 展示了聊天、
推理和安全偏好上的分项比较；项目至少保留按领域、长度和安全类别切片的验证集，并人工检查高分
但明显错误的回答。

## 4. PPO：toy objective 不等于 PPO 系统

[PPO](https://arxiv.org/abs/1707.06347) 的 clipped policy objective 使用：

```text
ratio = exp(logp_new - logp_old)
L_policy = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
```

`toy_ppo_clipped_loss` 只实现这段可单测的标量目标。完整 RLHF PPO 还需要：

1. 当前策略按 prompt 在线生成 response；
2. 奖励模型打分，并通常加入相对 reference policy 的 KL 约束；
3. value model 给出 token value；
4. 依据终止、padding 和 bootstrap 语义计算 returns/GAE；
5. 固定 rollout 的 old log-prob，分 minibatch、多 epoch 更新策略与 value；
6. 监控 KL、clip fraction、entropy、value error、奖励分解和生成长度。

[InstructGPT](https://arxiv.org/abs/2203.02155) 是 LLM 中 SFT—RM—PPO 组合的代表性一手来源。
只有 pairwise reward loss，或只有 clipped policy 公式，都不应命名为“完整 PPO”。

## 5. DPO：离线偏好优化也有序列契约

[DPO](https://arxiv.org/abs/2305.18290) 比较策略与参考模型在 chosen/rejected 上的相对
log-prob：

```text
margin = (logpi_chosen - logpi_rejected) - (logref_chosen - logref_rejected)
L_DPO = -log sigmoid(beta * margin)
```

实现前先锁定：

- 四组 log-prob 是否来自完全相同的模板、tokenizer 和 response mask；
- 使用序列 `sum` 还是长度归一化 `mean`；
- reference 是否冻结，adapter 开关是否正确；
- beta 的定义、reduction 和样本权重；
- chosen/rejected 是否共享 prompt 且截断一致。

总和形式会随回答长度累积，均值形式又改变原始目标；二者都可能产生长度偏差，不能静默切换。
`dpo_loss` 接收已经聚合的四组序列 log-prob，并严格拒绝空输入、非有限值和非法 beta。

## 6. GRPO、RLVR 与 2025–2026 的演化

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 将 GRPO 用于数学推理：对同一问题采样一组回答，
用组内奖励形成相对优势，省去独立 critic。`group_relative_advantages` 只实现组内标准化，
`toy_grpo_clipped_loss` 只把该优势接入 clipped ratio；它们都不是完整 GRPO。完整方法仍涉及组采样、
旧策略/reference 概率、KL、token mask、优化 epoch 和奖励设计。

RLVR（reinforcement learning with verifiable rewards）使用数学答案、程序测试或规则检查器提供奖励。
可验证不等于不可作弊：答案解析漏洞、弱单元测试、浮点容差、格式捷径和训练/评测泄漏都会造成
虚假高分。验证器应版本化，并维护“奖励通过但语义错误”的对抗样例库。

当前演化建议按一手论文阅读，而不是把它们当成已统一的工程共识：

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：推理 RL、冷启动和多阶段训练报告；
- [DAPO](https://arxiv.org/abs/2503.14476)：讨论动态采样、裁剪与稳定训练的 recipe；
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)：分析 GRPO 中可能引入优化偏差的归一化；
- [GSPO](https://arxiv.org/abs/2507.18071)：把重要性比率与裁剪提升到序列级目标。

这些论文的采样预算、模型版本、奖励实现和系统细节不同。复现实验必须同时记录每 prompt 样本数、
温度、最大生成长度、过滤规则、有效组比例、奖励均值/方差、KL 和实际训练 token 数。

## 7. 安全、红队与能力—安全联合评测

后训练验收不能只测 helpfulness。至少分别报告：

- helpfulness、事实性和指令遵循；
- harmlessness、危险内容边界、隐私与偏见；
- over-refusal 与 under-refusal；
- 越狱、提示注入、多语言和多轮攻击；
- reward model、训练中策略和最终 checkpoint 的相同题集对照。

[Constitutional AI](https://arxiv.org/abs/2212.08073) 是规则反馈和 harmlessness 训练的代表工作，
[Llama 2](https://arxiv.org/abs/2307.09288) 报告了安全 SFT、RLHF 与红队流程。课程把安全评测分为：

- development eval：训练中频繁运行，用于发现回归；
- assurance eval：隔离题集、低频运行，避免被反复调参污染；
- red-team set：持续加入真实失败模式，但保留来源、版本和披露边界。

训练数据或安全题的许可证、隐私信息和访问控制也属于模型治理。任何“安全提升”都应同时给出能力
退化和拒答率变化，不能只挑一个总分。

## 常见误区

- response-only SFT 是配置选择，不是所有模型唯一正确的训练方式。
- LoRA 降低可训练参数，不保证吞吐或峰值显存同比改善。
- 奖励模型的高准确率不代表偏好没有系统偏差。
- DPO 不需要在线 rollout，但仍依赖 reference、序列 log-prob 和数据质量。
- group-relative advantage、PPO clip 或一个 verifier 都不是完整训练系统。
- 推理能力可能来自数据、采样预算、验证器、基础模型和算法共同作用。

## 阶段验收

1. 运行 `uv run pytest -q tests/test_post_training.py`，覆盖 shift、mask、merge、严格输入和梯度。
2. 用手算样例核对 `sequence_logprob`、reward、DPO、toy PPO 和 toy GRPO 的符号与 reduction。
3. 保存 chat template、tokenizer、聚合方式、beta/KL、采样参数和奖励版本。
4. 对 SFT 前、SFT 后、偏好优化后 checkpoint 运行同一套能力、安全和拒答评测。
5. 报告 toy helper 的边界；只有补齐 rollout、reference/value、KL、mask 和训练循环后才使用完整算法名。

## 一手资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [LoRA](https://arxiv.org/abs/2106.09685)；[QLoRA](https://arxiv.org/abs/2305.14314)
- [PPO](https://arxiv.org/abs/1707.06347)；[DPO](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)；[DAPO](https://arxiv.org/abs/2503.14476)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)；[GSPO](https://arxiv.org/abs/2507.18071)
- [RewardBench](https://arxiv.org/abs/2403.13787)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)；[Llama 2](https://arxiv.org/abs/2307.09288)