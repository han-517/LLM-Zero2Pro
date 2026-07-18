# 论文库使用说明

- `catalog.yaml`：已经人工分级的课程论文。
- `inbox.yaml`：自动发现的候选项；候选不等于推荐。

`schema.json` 由运行时代码真实校验。来源类型区分论文、预印本、技术报告、模型卡、
官方博客、文档与代码仓库；可选字段记录 arXiv/DOI、数据、证据位置和核验状态。
- `schema.json`：`PaperRecord` 的公开数据契约。
- `notes/TEMPLATE.md`：三遍阅读笔记模板。

```powershell
uv run llm-course papers validate
uv run llm-course papers validate --check-links
uv run llm-course papers update --max-results 20
uv run llm-course papers graph
```
uv run llm-course papers update --profile data --since 2025-01-01
uv run llm-course papers update --profile training-systems
uv run llm-course papers update --profile attention
uv run llm-course papers update --profile moe
uv run llm-course papers update --profile posttraining

把候选项提升到 `catalog.yaml` 前，必须核对原始论文、版本日期、作者代码、证据类型、
局限性和重复 arXiv/DOI。普通提交只做离线 Schema 校验；网络链接检查由定时 CI 运行。

