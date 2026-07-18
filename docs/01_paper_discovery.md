# 多源论文发现器

`papers update` 的目标不是自动替你决定“什么值得学”，而是生成一个可追踪、可去重的候选池。正式课程目录与候选池严格分离。

## 三个来源

- arXiv：适合按主题和发布日期发现最新预印本。
- Semantic Scholar：补充 DOI、引用数、参考文献数和学科字段。公共接口高峰期可能返回 429，可选设置 `SEMANTIC_SCHOLAR_API_KEY`。
- Hugging Face Daily Papers：补充社区筛选后的近期论文及可能存在的官方代码链接。公开读取不要求令牌；设置 `HF_TOKEN` 只是可选项。

```powershell
# 尝试全部来源；任何单一来源失败都只产生警告
uv run llm-course papers update --source all --max-results 20

# 按课程域和日期检索；profile 还包括 evaluation-safety
uv run llm-course papers update --profile data --since 2025-01-01
uv run llm-course papers update --profile training-systems
uv run llm-course papers update --profile attention
uv run llm-course papers update --profile moe
uv run llm-course papers update --profile posttraining

# 单独调试某一来源
uv run llm-course papers update --source arxiv --max-results 10
uv run llm-course papers update --source semantic-scholar --max-results 10
uv run llm-course papers update --source huggingface --max-results 10

# 自定义 query 会覆盖 profile，但仍会真实传给上游来源
uv run llm-course papers update --source arxiv --query "linear attention language model"

# --since 使用 YYYY-MM-DD；非法日期会在联网前被拒绝
```

## 去重与人工闸门

更新器依次使用 arXiv ID、DOI、规范化标题去重，并同时排除已经进入 `catalog.yaml` 的论文。每条候选会保存来源、日期、摘要、引用统计、代码链接和发现日期。自动更新永远不会改变论文层级，也不会修改课程周次。

人工审核时至少回答：

1. 它解决的是架构、训练、数据、系统还是评测问题？
2. 技术主张由什么实验支持，有无强基线和消融？
3. 是否有官方实现或足够细节进行缩尺复现？
4. 它与目录中哪些论文构成 `builds_on`、`improves`、`contrasts_with` 或 `used_by`？
5. 是应升级为 Deep Dive，还是继续留在 Frontier Radar？

候选被人工升级后，应完整补齐 `PaperRecord`，运行 `papers validate`，再重新生成论文关系图。
