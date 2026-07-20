# 论文库使用说明

- `catalog.yaml`：已经人工分级并通过 Schema 校验的课程论文。
- `inbox.yaml`：自动发现的候选；候选不等于推荐。
- `schema.json`：论文、报告、模型卡、官方博客、文档和代码仓库的元数据契约。
- `notes/TEMPLATE.md`：三遍阅读笔记模板。

从仓库根目录运行：

```text
uv run llm-course papers validate
uv run llm-course papers validate --check-links
uv run llm-course papers update --max-results 20
uv run llm-course papers update --profile attention --since 2025-01-01
uv run llm-course papers graph
```

可用 profile 包括 `data`、`training-systems`、`attention`、`moe` 和 `posttraining`。把候选提升到 `catalog.yaml` 前，必须核对原始来源、版本日期、作者代码、证据类型、局限性和重复 arXiv/DOI。

普通提交只做离线 Schema 校验；网络链接检查由定时 CI 运行，避免临时网络故障阻断课程开发。