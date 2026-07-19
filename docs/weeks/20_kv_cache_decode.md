# 第 20 周：KV Cache——让逐 token 解码不再重算历史

## 课程定位

本周把前一周的共享 KV heads 放进真实生成循环。目标是区分 Prefill 与 Decode、设计明确 cache API，并用完整前缀重算作数值 oracle。KV Cache 是算法状态，不是无成本加速：它减少历史投影和 attention 子图重算，同时引入随 batch、层数和序列增长的持久内存。

## 学习目标

- 区分 Prefill 的并行计算与 Decode 的单步带宽路径。
- 定义每层 cache 的 shape、position、dtype、生命周期和返回策略。
- 证明逐 token cached logits 与完整前缀 logits 一致。
- 解释 cache 满、滑动窗口、绝对位置与 RoPE 重新编号的差别。

## 前置

需要掌握 MHA/GQA、causal bottom-right mask、RoPE position IDs 和自回归生成。应能解释为什么新 Query 长度为 1 时，Key 长度等于历史加当前 token，以及为什么训练时通常不需要把整层 K/V 作为 Python 返回值保存。

## 直觉

没有 cache 时，生成第 `t` 个 token 会再次为前 `t-1` 个 token 计算每层 K/V，早先工作被反复重做。cache 把每层历史 K/V 留下来，新一步只投影当前 token 的 Q/K/V，把新 K/V 追加，再让新 Q 查询全部历史。它像一本不断增页的索引：查得更快，但书本占内存并且必须跟位置编号保持一致。

## 张量/数据契约

每层 cache 是 `(K,V)`；GQA 下 `K:[B,Hkv,Tpast,Dh]`、`V:[B,Hkv,Tpast,Dv]`。当前输入 `x:[B,Tnew,D]`，默认 position IDs 为 `[Tpast,...,Tpast+Tnew-1]`。追加后 key length 为 `Tpast+Tnew`，causal mask 把当前 Query 对齐到 Key 序列右下角。所有层 cache 数量等于层数、batch/head/head_dim 相容、历史长度一致。训练调用可 `return_caches=False`，避免无意保留大状态。

## 推导与机制

完整重算在第 `t` 步会再次执行历史 token 的 QKV projection 和各层 block；累计工作随生成长度快速增长。cache 后，历史 K/V projection 不再重复，但 attention 仍需读取长度 `t` 的历史 K/V，所以 decode 每步工作不是常数。持久缓存字节为

\[
B L T 2 H_{kv}D_h s.
\]

FlashAttention 改善 attention 中间 IO，不会消除这份持久状态；PagedAttention 改善分配和共享，也不改变内容大小公式。滑窗只保留最近 `W` 个 K/V，才会把状态上界截断，但它改变可见连接。

## 数值例

一个两层、`B=1,Hkv=2,Dh=4` 的 FP32 toy，处理 5 个 token 后每层 K 与 V 各有 `1×2×5×4=40` 元素，两者 320 字节，两层 640 字节。第 6 个 token 的 Query shape 是 `[1,Hq,1,4]`，K 长度为 6；bottom-right causal mask 对这唯一 Query 允许读取 0..5，而不是只读第 0 个位置。

## 最小代码

```python
import torch


def append_kv(new_k, new_v, cache=None):
    if new_k.shape[:-1] != new_v.shape[:-1]:
        raise ValueError("new K/V 的 batch、head、time 必须一致")
    if cache is None:
        return new_k, new_v
    old_k, old_v = cache
    if old_k.shape[:2] != new_k.shape[:2] or old_k.shape[-1] != new_k.shape[-1]:
        raise ValueError("cache K 契约不匹配")
    if old_v.shape[:2] != new_v.shape[:2] or old_v.shape[-1] != new_v.shape[-1]:
        raise ValueError("cache V 契约不匹配")
    return (
        torch.cat((old_k, new_k), dim=-2),
        torch.cat((old_v, new_v), dim=-2),
    )


first_k = torch.arange(16.0).reshape(1, 2, 2, 4)
first_v = -first_k
cache = append_kv(first_k, first_v)
next_k = torch.full((1, 2, 1, 4), 99.0)
next_v = -next_k
full_k, full_v = append_kv(next_k, next_v, cache)
assert full_k.shape == (1, 2, 3, 4)
torch.testing.assert_close(full_k[..., :2, :], first_k)
torch.testing.assert_close(full_v[..., -1:, :], next_v)
```
课程用 `torch.cat` 追求可读性，会每步重新分配并复制 metadata/数据；生产 runtime 使用预分配、paged blocks 或专门 cache manager。课程结果用于 correctness，不代表最佳 decode allocator。

## 反例与调试

若把未旋转 K 写入 RoPE cache，历史位置在下一步无法恢复；若每步 position ID 都从 0 开始，cached/full 立即分叉。普通上三角 mask 在 `Tq=1,Tk>1` 时常只允许第一个 Key，必须用 bottom-right 对齐。生成窗口满后直接丢最老 cache 却沿用学习式绝对位置 embedding，会与重建窗口语义不同。比较速度前先检查输出；一个错误地只看最后 token 的实现可能很快但没有意义。

## 主流工作与证据等级

Transformer 解码与 MQA/GQA 论文提供算法基础；vLLM/PagedAttention 论文和官方设计展示生产系统如何管理动态 cache。Hugging Face Transformers 的 cache 文档是常用官方 API 证据。不同 runtime 对 static、dynamic、sliding、quantized cache 的支持变化很快，教程只固定概念契约，具体参数需查当前官方文档。

## Notebook、互动图与 starter

使用 `docs/interactive/core-concepts.html#kv-cache` 和 `docs/interactive/architecture-lab.html` 观察长度、层数、KV heads；完成 `exercises/starter/03_kv_cache_budget.py`；在 `notebooks/core/06_modern_decoder.ipynb` 逐 token 比较 logits。互动图的连续数组不展示分页碎片，分页留到推理阶段。

## 实验

覆盖 MHA/GQA、绝对位置/RoPE、一次多个新 token 与单 token、不同 batch、窗口重建。每次记录最大 logits 差、cache shape 和实际字节。墙钟基准分别测 Prefill 与 Decode，预热后报告中位数，并注明课程 `torch.cat` 的分配开销。再故意使用错误 mask，保存失败样例作为调试文档。

## 验收 rubric

- 40%：full/cached logits 在所有配置下容差一致。
- 20%：position、mask、层数与 cache shape 契约完整。
- 25%：资源报告区分 Prefill、Decode、持久内存和临时激活。
- 15%：明确教学 `torch.cat`、paged runtime 与滑窗语义边界。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Hugging Face KV Cache 官方说明](https://huggingface.co/docs/transformers/kv_cache)
