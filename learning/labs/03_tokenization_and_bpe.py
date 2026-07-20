# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown] llm_course_preamble=true
# **课程契约** · 周次 9, 10 · 预计 90 分钟 · Starter 06 · 默认 CPU/离线。

# %% [markdown]
# # Byte BPE：字符、字节与 token
#
# 先观察同一字符串的字符数与 UTF-8 字节数，再训练确定性的教学版 Byte BPE。

# %%
from llm_from_scratch.tokenization import BytePairTokenizer

text = "LLM 学习：hello hello"
raw = list(text.encode("utf-8"))
tokenizer = BytePairTokenizer.train(text * 4, vocab_size=264)
ids = tokenizer.encode(text)
print({"characters": len(text), "bytes": len(raw), "tokens": len(ids)})
assert tokenizer.decode(ids) == text

# %% [markdown]
# ## 练习
#
# 打开 learning/labs/starter/06_byte_bpe.py。核心 merge 逻辑保持空缺；完成后运行：
#
#     uv run llm-course exercises check 06

# %% [markdown] llm_course_enrichment=true
# ## 学习目标与形状账本
#
# 完成后你应能区分 Unicode code point、UTF-8 byte 与 token id；手推一次 BPE merge；解释词表大小、序列长度和未知字符处理之间的权衡。
#
# `str → bytes: [N_bytes] → token ids: [T] → embedding: [T, d_model]`。

# %% llm_course_enrichment=true
samples = ["A", "汉", "🙂", "é"]
for item in samples:
    raw = item.encode("utf-8")
    print(repr(item), "字符数=", len(item), "字节=", list(raw), "字节数=", len(raw))

# %% [markdown] llm_course_enrichment=true
# ## 1. Byte-level BPE 的训练循环
#
# 初始词表是 256 个单字节。每轮统计相邻 token pair，选出频次最高的一对并赋予新 id，然后重写序列。编码时必须按训练出的 merge 顺序执行。

# %% llm_course_enrichment=true
from llm_from_scratch.tokenization import BytePairTokenizer

corpus = "banana bandana banana"
tok_a = BytePairTokenizer.train(corpus, vocab_size=262)
tok_b = BytePairTokenizer.train(corpus, vocab_size=262)
assert tok_a.to_dict() == tok_b.to_dict()
print([(m.pair, m.token_id) for m in tok_a.merges])

# %% [markdown] llm_course_enrichment=true
# ## 2. 压缩率不是唯一目标
#
# 更大词表通常缩短序列，但会增大 embedding/lm-head 参数，并可能把低频形态切得不稳定。跨 tokenizer 比较 perplexity 也不公平，因为一个 token 承载的信息量不同。

# %% llm_course_enrichment=true
text = "banana bandana"
base_tokens = list(text.encode("utf-8"))
learned_tokens = tok_a.encode(text)
assert tok_a.decode(learned_tokens) == text
print({"byte_tokens": len(base_tokens), "bpe_tokens": len(learned_tokens), "ids": learned_tokens})

# %% [markdown] llm_course_enrichment=true
# ## 3. 序列化契约
#
# Tokenizer 是模型输入协议的一部分。保存模型时还要保存 merge 表、特殊 token、规范化规则和版本；同名词表文件不足以证明协议一致。

# %% llm_course_enrichment=true
state = tok_a.to_dict()
restored = BytePairTokenizer.from_dict(state)
assert restored.encode(text) == learned_tokens
try:
    restored.decode([9999])
except ValueError as error:
    print("预期错误：", error)

# %% [markdown] llm_course_enrichment=true
# ## 练习与验收
#
# 填写 `../../learning/labs/starter/06_byte_bpe.py`，运行 `uv run llm-course exercises check 06`。验收：中英混合和 emoji 往返编码；merge 确定；未知 id 明确报错；报告 bytes/token。

# %% [markdown] llm_course_enrichment=true
# ## 一手来源
#
# [BPE](https://arxiv.org/abs/1508.07909)、[SentencePiece](https://arxiv.org/abs/1808.06226)、[GPT-2 byte-level BPE](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。本 Notebook 不含生产 tokenizer 的全部 normalization、pre-tokenization 与特殊 token 细节。
