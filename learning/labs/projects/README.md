# 贯穿式大作业

这里不是第二套课程路线。仍按 `learning/README.md` 的第 1–48 课学习；小型
`starter/` 用来练单个函数，大作业用来证明你能把多个知识点连接成可运行系统。

| 编号 | 相关课次 | 状态 | 目标 |
|---:|---:|---|---|
| 01 | 9–21 | 可开始 | 从字节 BPE 到训练、恢复与生成的完整语言模型 |
| 02 | 22、23、28 | 可开始 | 真实网页文档清洗、审计与数据消融 |
| 03 | 25、26、29 | 可开始 | GPU profiling、Triton、DDP 与分片训练 |
| 04 | 27、28 | 可开始 | 受预算约束的 Scaling Law 实验 |
| 05 | 40–44 | 可开始 | SFT、rollout、可验证奖励与策略更新 |

查看状态：

```text
uv run llm-course projects list
```

可用大作业的公开核查（未填写核心代码时失败是正常学习反馈）：

```text
uv run llm-course projects check 01
uv run llm-course projects check 02
uv run llm-course projects check 03
uv run llm-course projects check 04
uv run llm-course projects check 05
```

五项大作业均已提供 learner-owned starter 和公开核查。必须继续完成各 README 中的实际运行产物与报告；公开测试只覆盖最低行为契约。
