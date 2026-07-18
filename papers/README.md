# 论文库使用说明

- `catalog.yaml`：已经人工分级的课程论文。
- `inbox.yaml`：自动发现的候选项；候选不等于推荐。
- `schema.json`：`PaperRecord` 的公开数据契约。
- `notes/TEMPLATE.md`：三遍阅读笔记模板。

```powershell
uv run llm-course papers validate
uv run llm-course papers validate --check-links
uv run llm-course papers update --max-results 20
uv run llm-course papers graph
```

把候选项提升到 `catalog.yaml` 前，必须核对原始论文、版本日期、作者代码、证据类型、局限性和重复 arXiv/DOI。

