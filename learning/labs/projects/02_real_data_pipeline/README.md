# 大作业 02：从网页文档到可审计预训练数据

本项目把第 22、23、28 课从“阅读过滤规则”升级为可运行的数据供应链。你要维护
`student_pipeline`，处理仓库内的合成 HTML 快照，不下载真实网页，也不把教学启发式
冒充生产语言识别、PII 检测、内容安全或法律判断。

## 前置课次

完成第 22、23、28 课与 starter 14，并先完成大作业 01，以便最后做同预算数据消融。

## 固定顺序

1. **抽取**：只保留可见文本，忽略 script/style/template，解码实体并稳定化空白。
2. **治理与隐私**：保留来源、快照、许可和 SHA-256；隔离许可未知项；脱敏邮箱与 API key，审计日志不得保存原秘密。
3. **语言与质量**：实现仅覆盖英文/中文的脚本启发式与显式 `unknown`；记录字符、字母、重复行和词长指标。
4. **近重复**：用稳定哈希实现 character-shingle MinHash、传递聚类和 canonical group；禁止使用进程随机化的 `hash()`。
5. **切分与审计**：按 duplicate group 切分，确保家族不跨 train/validation；每条原始记录必须有 terminal decision。
6. **消融**：在相同 tokenizer、模型、token 预算和 seed 下比较 raw baseline 与 filtered 数据。

## 文件与产物

```text
student_pipeline/pipeline.py  # 你填写的核心管线
run_pipeline.py               # 离线 JSONL → processed/audit/data card
data/raw_documents.jsonl      # 合成网页快照；含近重复、许可、PII 和低质量反例
report_template.md            # 漏斗、抽检、误差和数据消融报告
artifacts/processed.jsonl     # 接受的文本、group 与 split
artifacts/audit.jsonl         # 所有阶段的可追溯事件
artifacts/data_card.json      # 文档/字符两套保留统计
```

实际目录名是 `data/`；上面的中文说明只用于解释职责。

## 核查与运行

```text
uv run llm-course projects check 02
uv run python learning/labs/projects/02_real_data_pipeline/run_pipeline.py
```

## 完成标准

- 公开核查通过；同一输入跨进程产生相同 MinHash、duplicate group 与 split。
- 原始文档数等于 accepted + rejected；无静默丢失，字符保留率与文档保留率分开。
- audit 事件只存敏感片段摘要，不泄露原邮箱或密钥。
- 抽检每类拒绝原因并报告误删、漏删；解释 toy detector 对混合语言、混淆字符和代码的失败。
- 同预算消融保留原始运行记录，不能只报告过滤后数据更“干净”。

生产系统还需要成熟 HTML/WARC 解析、训练或校准的语言/安全分类器、分布式 LSH、删除传播、访问控制、人工复核与法律流程；本项目只验证这些接口为何必须存在。