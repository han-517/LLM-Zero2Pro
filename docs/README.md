# 学习入口

现在只有一条[统一学习路线（1–48）](learning_path.md)。请先从这条路线找到当前周，再进入对应材料。

- [统一学习路线](learning_path.md)：决定当前周读什么、运行什么、提交什么。
- [逐周教材](weeks/)：48 个学习单元，每周一篇。
- [交互实验](interactive/)：观察 RoPE、MoE 等机制。
- [架构演化](architecture_evolution.md)：用于串联概念，不是另一条路线。
- [扩展内容](extensions/)：完成主线后的补充。

## 每周流程

1. 找到当前周。
2. 阅读对应的 `weeks/NN_*.md`。
3. 运行对应 Notebook 并完成练习。
4. 重启后复现，更新进度。

## 修改课程后

运行 `uv run llm-course course path --write`。它只维护 `docs/learning_path.md`，不会再生成两个学习路线。
