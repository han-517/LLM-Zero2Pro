# 第 30 周：滑窗、块稀疏与全局 token

## 课程定位

FlashAttention 保持所有连接，本周开始真正改变 Query-Key 图。滑窗把每个 token 的直接连接限制在局部，块稀疏用规则块提高 kernel 友好性，全局 token 或检索 indexer 补回远距离路径。目标不仅是画 mask，还要分析多层可达性并构造“被断开的必要依赖”反例。

## 学习目标

- 实现 full/cached bottom-right sliding-window mask。
- 区分 local、dilated、block、global+local 与 learned retrieval sparse。
- 推导连接数和多层理论感受野，并识别信息瓶颈。
- 用远距离复制/检索任务同时测质量与稀疏成本。

## 前置

需要掌握图可达性、causal mask、cached decode 与 attention shape。应能区分“理论非零连接数”与“规则稀疏 kernel 实际加速”，因为任意 bool mask 在 dense kernel 上仍可能计算完整矩阵。

## 直觉

局部窗口像每个人只和附近同学传话；一层只能看到近邻，多层后消息逐步传播，但路径变长。全局 token 像公告栏，可用两跳连接远处；dilation 像隔几个座位建立快速通道；learned retrieval 像先用轻量索引器挑可能相关的远处资料。连接更少意味着便宜，也意味着可能把答案所在位置彻底排除。

## 张量/数据契约

`sliding_window_mask(Tq,window,key_length=Tk)` 返回 `[Tq,Tk]` bool，`True` 表示允许。Query 位置按 `Tk-Tq...Tk-1` 对齐，条件为 `k<=q` 且 `k>=q-window+1`。加入全局 token 后，需要定义它是否双向可见以及因果生成中能否读取未来；不能照搬双向编码器规则。块稀疏要求 block size、尾块和布局 metadata。检索式 top-k 还需 index scores/indices 与 recall 评测。

## 推导与机制

单层 causal window `w` 的连接约 `T·w`，而 full 为约 `T²/2`。若每层只向后看 `w-1` 个额外位置，`L` 层末端 token 的理论连续感受野上界约 `1+L(w-1)`，不是 `w^L`。全局 token 可缩短图距离，但单个 bottleneck 的容量有限。块稀疏若每个 Query block 连接 `k` 个 Key blocks，连接元素约 `T·k·block_size`。实际复杂度还受 indexer 和不规则 gather 支配。

## 数值例

序列 64、窗口 4、3 层，末端 token 最远可通过连续局部路径到约 `64-(1+3·3)+1=55` 附近，即覆盖约 10 个位置，而非全序列。cached decode `Tq=1,Tk=5,w=2` 的 mask 应为 `[False,False,False,True,True]`。若错误按局部 Query index 0 构造，会得到 `[True,False,...]`。

## 最小代码

```python
def sliding_window_mask(query_length, key_length, window, device=None):
    if not (1 <= query_length <= key_length) or window < 1:
        raise ValueError("非法长度或窗口")
    q = torch.arange(key_length - query_length, key_length, device=device)[:, None]
    k = torch.arange(key_length, device=device)[None, :]
    return (k <= q) & (k >= q - window + 1)
```

这段只生成 mask。若仍把它喂给 dense attention，内存/计算不一定按 `T·w` 降低；生产加速需要窗口专用或块稀疏 kernel、布局和调度。Longformer 的双向 encoder 规则也不能原样当 causal decoder。

## 反例与调试

第一类错位是 cached Query 从 0 编号。第二类是窗口定义到底含当前 token 还是“前 w 个历史”，差 1 会改变连接数。第三类是把全局 token 设为可读取所有未来位置，破坏因果性。用未来 K/V 扰动测试因果；用唯一答案放在窗口外测试断路；用多层邻接矩阵布尔乘验证可达性。性能无提升时检查 kernel 是否仍 dense。

## 主流工作与证据等级

Longformer、BigBird 给出局部/全局/随机稀疏基础实验；Mistral 等公开模型采用 sliding window，属于模型采用证据。DeepSeek-V3.2 DSA 使用 learned indexer 选择 top-k KV，属于 2025 技术报告与官方代码证据，其训练续接和 MLA 基座不可省略。规则稀疏成熟度高于新近 learned sparse，二者应分级。

## Notebook、互动图与 starter

在 `learning/readings/interactive/architecture-lab.html` 调窗口与层数；使用 `learning/labs/08_attention_frontiers.ipynb` 画 mask、邻接和多层可达性；完成 starter `16` 的 sliding-window 部分。互动条形图是上界直觉，正式实验用图算法核查。

## 实验

实现 full、window、dilated、global+local 四种 causal mask，覆盖 full/cached 与尾块。构造局部语言建模、远距离复制、多 key 检索三类任务，固定参数和训练 token。报告连接数、dense/专用 kernel 墙钟（若有）、验证 loss、按距离 recall。至少展示一个窗口彻底失败而 full 成功的反例。

## 验收 rubric

- 35%：mask、因果性、cached 对齐和边界测试正确。
- 25%：多层感受野/图可达推导与程序吻合。
- 25%：质量、连接数和真实 kernel 成本分别报告。
- 15%：区分规则 mask、专用 kernel 与 learned indexer 证据。

## 一手来源

- [Longformer](https://arxiv.org/abs/2004.05150)
- [Big Bird](https://arxiv.org/abs/2007.14062)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [DeepSeek-V3.2 / DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556)
