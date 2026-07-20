# 第 22 周：预训练数据治理——来源、许可与可追溯数据卡

## 课程定位

模型结构之外，预训练最不可逆的决定往往是“哪些数据进入了训练”。本周不把互联网文本视为无主的字符串，而是建立来源、快照、许可、隐私、删除与污染的可审计记录。课程不提供法律意见；它训练工程证据链，使研究者能回答每个 token 从哪里来、经过什么处理、为什么可以进入当前实验。

## 学习目标

- 设计数据卡与 provenance manifest，记录来源、版本、许可和处理代码。
- 区分公开可访问、许可允许、合理使用判断、隐私与 robots 等不同问题。
- 为 PII、密钥、有害内容、恶意代码和评测污染设计分开的控制与审计。
- 用小型语料生成可复现快照，而非只保存处理后的单一文本文件。

## 前置

需要会使用哈希、JSON/YAML、tokenizer 和 train/validation split。应理解 SHA-256 证明“字节相同”，不能证明许可有效；随机种子保证抽样可重放，也不能替代来源记录。

## 直觉

把数据管道看作带版本的供应链。原始对象是货物，来源和许可像供应商凭证，清洗规则像加工工序，tokenizer 版本像计量标准。只保留最终 `train.bin` 等于撕掉所有标签：即使模型效果好，也无法回应污染、删除或分布偏差。治理不是训练完成后的文档工作，而是管道输入输出都必须携带的元数据。

## 张量/数据契约

原始记录至少包含 `document_id、source_uri、snapshot_time、content_hash、license_status、language、raw_bytes`。处理事件包含 `rule_id、rule_version、decision、reason、input_hash、output_hash`。划分记录包含 split、seed 与 group key，防止同一站点或文档家族跨 train/eval。token 产物需记录 tokenizer hash、EOS 规则与 token 数。空文档、解码失败和被拒绝记录也要计数，不能静默消失。

## 推导与机制

若管道有阶段 `S_0...S_n`，每阶段保留率 `r_i=N_i/N_{i-1}`，总文档保留率为 `∏r_i`；token 保留率要单独计算，因为过滤短文与长文对 token 分布影响不同。分层来源混合概率 `p_s` 与来源内抽样 `P(d|s)` 决定训练分布 `P(d)=p_sP(d|s)`。数据卡必须同时记录原始占比、采样占比和训练实际 token 占比，避免把“文件数占比”误当“token 占比”。

## 数值例

某来源有 100 万文档、20 亿 token。许可筛查拒绝 5%，PII/密钥规则拒绝剩余的 2%，语言与质量规则再拒绝 15%，文档保留约 `0.95×0.98×0.85=79.1%`。若被拒绝的多为长文，token 保留可能只有 65%。只报告“保留 79% 文档”会误导实际训练组成。若混合时把该来源上采样 2 倍，最终 token 比例还会再次改变。

## 最小代码

```python
from dataclasses import asdict, dataclass
from hashlib import sha256

@dataclass(frozen=True)
class Provenance:
    document_id: str
    source_uri: str
    snapshot: str
    license_status: str
    content_sha256: str

def make_record(document_id, source_uri, snapshot, license_status, raw):
    digest = sha256(raw).hexdigest()
    return asdict(Provenance(document_id, source_uri, snapshot, license_status, digest))
```

这是本地 manifest baseline。生产数据湖还需要不可变对象存储、访问控制、删除传播、审计日志、schema 演化和跨地区政策；课程代码不能判断版权或隐私合法性。

## 反例与调试

把 `source="web"` 写进所有记录不算 provenance。URL 可变，必须有抓取或发布快照与内容哈希。把许可未知当成开放许可是危险默认值，应进入隔离队列。只删除处理后文本却保留所有中间副本，无法满足删除传播。若 train/eval 先逐文档随机切分再做去重，近重复家族会跨 split；应先确定 group/去重策略。过滤器版本变更后若仍复用旧 cache，统计会与代码不一致。

## 主流工作与证据等级

Datasheets for Datasets 提供系统文档框架，属于治理方法证据。Dolma、FineWeb、DataComp-LM 公开数据流程、消融与部分工具，属于可审计开放语料案例；它们的规则是特定目标下的工程选择，不是普适法律标准。公开模型报告只给来源类别而没有可重放清单时，证据等级低于完整开放管道。

## Notebook、互动图与 starter

在 `learning/readings/interactive/training-and-alignment.html` 调节过滤漏斗；使用 `learning/labs/07_pretraining_systems.ipynb` 生成来源、语言、长度和许可状态统计；完成 starter `14` 的 data card/manifest 部分。Notebook 只处理课程小数据，不能声称覆盖互联网规模治理。

## 实验

选择三个许可状态不同的小语料源，保存不可变 raw snapshot、manifest 和处理事件。执行两版规则，比较文档/token 保留率、长度与语言分布，并随机抽检每种拒绝原因。再模拟一条删除请求，列出 raw、processed、packed、checkpoint 训练清单中需要更新的对象。

## 验收 rubric

- 30%：来源、快照、哈希、许可和规则版本可追溯。
- 25%：文档/token 两套漏斗统计完整且无静默丢失。
- 25%：污染、隐私、许可、内容安全分别建模。
- 20%：明确课程 manifest 与生产治理/法律判断边界。

## 一手来源

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159)
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [DataComp-LM](https://arxiv.org/abs/2406.11794)
