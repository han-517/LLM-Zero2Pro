# 大作业 05 后训练与 RLVR 报告

## 数据与 mask 契约

- prompt/response 边界、EOS、padding 与被计分 token：
- SFT train/validation 切分和污染检查：
- 每个 prompt 的 rollout 数、温度、seed 与生成长度：

## SFT

报告初末 loss、梯度范数、held-out exact match 和至少一个格式扰动测试。确认 prompt token 不进入 response-only loss。

## RLVR / GRPO

| step | reward mean/std | loss | approximate KL | grad norm | exact match |
|---:|---:|---:|---:|---:|---:|
| | | | | | |

说明 token-level ratio、response-length normalization、population std、zero-variance group、KL estimator 和 reference policy 的具体选择。不要把 toy objective 写成完整 PPO/GRPO 训练框架。

## 奖励黑客与安全边界

- 构造至少两个“形式正确但语义错误”或利用 verifier 漏洞的输出。
- 用未见 prompt、长度变化和格式变化复测；保留所有原始 rollout。
- 区分任务准确率、偏好对齐、拒答行为和广义安全；本项目只直接测第一项。
