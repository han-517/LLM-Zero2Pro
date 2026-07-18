# 结构化知识区

`concepts.yaml` 记录概念之间的先修和权衡关系；论文之间的证据关系由下面的命令生成：

```powershell
uv run llm-course papers graph
```

生成结果是 `paper_graph.md`，可由支持 Mermaid 的 Markdown 阅读器显示。不要把图中的连线理解为“后者一定更好”；`improves` 只表示论文声称在指定设置下改进了某项指标，仍需回到证据和局限字段。

