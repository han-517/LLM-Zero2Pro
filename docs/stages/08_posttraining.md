# 阶段八：后训练——从“续写文本”到“按要求回答”

## SFT

监督微调仍是 next-token prediction，只是数据变成指令和回答。通常 prompt 用于提供上下文，但 loss 只计算 answer token：

```text
tokens:     [system, user, assistant answer]
loss_mask:  [0,      0,    1, 1, 1, ...]
```

模板、特殊 token 和 mask 错一位都会让模型学错目标。

还要处理 next-token shift：`logits[:, t]` 预测 `tokens[:, t+1]`，因此 answer mask 必须与
labels 一起右移，不能只截 logits。参考 API `causal_sft_loss(logits, token_ids, assistant_mask)`
负责同步移动三者；`masked_sft_loss` 只接收已经对齐的数据。先完成
`exercises/starter/04_sft_shift.py`，再构造“只改变 prompt logits，loss 不变”的反例测试。


## LoRA 与 QLoRA

冻结原权重 `W`，只学习低秩修正：

```text
W' = W + scale * B A
A: [rank, in], B: [out, rank]
```

rank 远小于输入/输出维时，训练参数大幅减少。QLoRA 再把冻结底座量化，但计算时需要合适的反量化和高精度累积。课程核心实现 LoRA；QLoRA 重点理解数据流。

## 偏好与奖励

成对偏好数据给出 chosen 和 rejected。奖励模型希望 `r(chosen) > r(rejected)`，常用 logistic pairwise loss。奖励只是人类偏好的代理，优化过度会产生 reward hacking。

## PPO

PPO 先由策略模型采样回答，再用奖励、价值模型和旧策略概率估计优势。核心概率比为
`ratio = exp(logp_new - logp_old)`；clipped objective 比较 `ratio * advantage` 与
`clip(ratio, 1-eps, 1+eps) * advantage`，限制一次更新偏离旧策略过远。LLM 对齐还常加入
相对参考模型的 KL 惩罚。

PPO 不是一个 pairwise loss 函数，而是一条“采样—打分—优势估计—多轮策略更新”的在线
训练流水线。本课程把完整 PPO 保持为概念和论文阅读项，不把单个奖励损失误称为 PPO 实现。

## DPO

DPO 比较策略模型和参考模型对 chosen/rejected 的 log-prob 差，不显式训练奖励模型。教学代码直接接收四组序列 log-prob，先确保符号和 beta 正确，再连接语言模型。

## GRPO 与 RLVR

GRPO 对同一问题采样一组回答，用组内均值和标准差形成相对优势。RLVR 使用可验证奖励，例如数学答案或单元测试。可验证不代表不可作弊：格式漏洞、测试不完整和数据泄漏仍会产生虚假高奖励。

## 常见误区

- SFT 不是把 prompt 和 answer 全部无差别算 loss。
- LoRA 减少可训练参数，不一定按同比例减少激活显存。
- 偏好数据代表收集协议，不是绝对真理。
- 强推理表现可能来自数据、采样预算、验证器和训练方法共同作用，不能只归因于算法名。

## 阶段验收

实现 answer mask、LoRALinear、pairwise reward loss、DPO loss 和 group-relative advantage；每个函数有极端输入和梯度测试。

