# 贯穿式大作业

这里不是第二套课程路线。仍按 `learning/README.md` 的第 1–48 课学习；小型
`starter/` 用来练单个函数，大作业用来证明你能把多个知识点连接成可运行系统。

| 编号 | 相关课次 | 状态 | 目标 |
|---:|---:|---|---|
| 01 | 9–21 | 可开始 | 从字节 BPE 到训练、恢复与生成的完整语言模型 |
| 02 | 22、23、28 | 建设中 | 真实网页文档清洗、审计与数据消融 |
| 03 | 25、26、29 | 建设中 | GPU profiling、Triton、DDP 与分片训练 |
| 04 | 27、28 | 建设中 | 受预算约束的 Scaling Law 实验 |
| 05 | 40–44 | 建设中 | SFT、rollout、可验证奖励与策略更新 |

查看状态：

```text
uv run llm-course projects list
```

大作业 01 的公开核查：

```text
uv run llm-course projects check 01
```

建设中表示规格已经固定，但 starter/checker 尚未启用，不计入当前毕业验收。
