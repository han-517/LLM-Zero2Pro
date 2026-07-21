# 大作业 05：SFT、grouped rollout 与可验证奖励策略更新

本项目把第 40–44 课从“计算几个 loss 数字”推进到真实 optimizer step：先做 response-only SFT，再由当前 policy 生成 grouped rollout，保存 behavior log-prob，用确定性 verifier 计算奖励，最后执行带 reference KL 的 GRPO 风格 RLVR 更新。

## 前置课次

完成第 40–44 课、starter 04/19 和大作业 01。该项目只使用合成 successor task，不下载模型权重，也不把玩具精确匹配等同于人类偏好或安全对齐。

## 固定顺序

1. **mask/shift**：`response_mask[:, t]` 表示目标 token `t` 是否进入 loss；prompt、padding 和 token 0 不计分。
2. **SFT step**：forward → response-only CE → finite gradient → clip → optimizer step，并在 held-out prompt 上评测。
3. **rollout**：每个 prompt 采样 G 个 response，保存生成时的 old token log-prob、EOS 后 mask、温度和 seed。
4. **reward/advantage**：verifier 只检查第一个 response token；组内使用 population std，零方差组优势必须为零。
5. **GRPO/RLVR step**：重新计算 current/reference log-prob，使用 sign-aware clipped surrogate、非负 KL estimator 和 response-length normalization 更新 policy；reference 不得变化。
6. **压力测试**：构造格式变化、长度变化和 verifier 漏洞，保留所有失败 rollout。

## 核查与运行

```text
uv run llm-course projects check 05
uv run python learning/labs/projects/05_alignment_rl/run_alignment.py
```

运行后生成 `artifacts/policy.pt` 和 `artifacts/metrics.json`。报告模板要求逐步 reward、KL、梯度范数和 exact match，而不是只展示最后一个成功样例。

## 完成标准

- 公开核查通过；prompt token 对 SFT/GRPO loss 没有贡献，空 response mask 明确报错。
- 相同 seed 的 rollout、old log-prob 和奖励可复现；reference 参数在更新前后逐元素不变。
- SFT 与 RLVR 都发生真实参数更新，loss/reward/KL/grad norm 有限并保存完整曲线。
- 至少报告两个 reward-hacking 或 distribution-shift 失败案例，并区分任务准确率、偏好对齐与安全。
- 明确本实现省略分布式 rollout engine、异步采样、经验回放、PPO value head、生产 verifier 与大规模安全评测。