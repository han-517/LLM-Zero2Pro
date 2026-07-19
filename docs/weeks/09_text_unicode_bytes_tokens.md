# 第 9 周：文本、Unicode、字节与 token

## 课程定位

语言模型不直接接收“文字”，而接收有限词表中的整数。本周建立从用户看到的字形到 Unicode code point、UTF-8 字节、token piece 和 token id 的完整边界。它是后续 BPE、embedding、数据清洗和上下文预算的地基：若把 Python 的字符数当字节数，或把 token 解码当普通字符串拼接，模型即使数学正确也会丢数据。目标不是记编码表，而是建立可往返、可版本化、可测试的文本契约。

## 学习目标

你应能解释字符、字素簇、code point、code unit、byte、token 的差异；手工拆解中文与 emoji 的 UTF-8；构造确定性字符/字节 tokenizer；验证 `decode(encode(text)) == text`；说明 normalization 会在何时改变原始字节；报告 bytes/token 而不是笼统说某语言“更耗 token”。还要知道 tokenizer 是模型接口的一部分，换词表后旧权重的第 $i$ 行不再具有相同语义。

## 前置知识与资产

只需 Python 的 `str`、`bytes`、字典和基础张量。主实验是 `notebooks/core/03_tokenization_and_bpe.ipynb`；本周是研究周，没有必填 starter，先不要提前实现第 10 周的 merge。互动入口 `docs/interactive/index.html` 用来查看 token 粒度变化；最可靠的核查仍是标准库 `encode/decode`、十六进制打印和往返断言。

## 自洽直觉

屏幕上的“é”可能是一个 code point `U+00E9`，也可能是 `e` 加组合重音两个 code point；家庭 emoji 还可能由多个 emoji、零宽连接符和修饰符组成，却显示成一个字素簇。UTF-8 再把每个 Unicode scalar value 变成 1–4 个 byte。tokenizer 最终把 byte 或字符片段映射为离散 id。于是同一个可见字符串经过不同 normalization 或词表可得到不同 id；反过来，单个 token 的 byte 序列也未必能单独解成合法 UTF-8，必须在拼接 byte 后统一解码。

## 张量/数据契约

原始输入是 Python `str`；显式选择 UTF-8 后得到 `bytes[N]`，每项为 0–255。byte tokenizer 的基本词表至少能表示 256 个 byte，`encode` 返回 `list[int]` 或 `LongTensor[T]`，每个 id 满足 `0 <= id < V`；`decode` 必须先把 piece 还原并连接为 bytes，再以声明的错误策略解码。批处理 token id 为 `[B,T]`，padding id、BOS、EOS 等特殊 id 必须与普通 byte token 不冲突，同时 attention mask 为 `[B,T]` 布尔值。数据契约应保存 normalization 形式、预分词规则、词表、merge 顺序、special-token 映射和版本哈希。

## 推导/机制：UTF-8 与词表映射

UTF-8 保持 ASCII 透明：`U+0000..U+007F` 用一个 byte。以“中”的 code point $U+4E2D$ 为例，数值落在三 byte 范围，将二进制位填入模板 `1110xxxx 10xxxxxx 10xxxxxx`，得到 `E4 B8 AD`。tokenizer 再实现函数 $E:\mathcal{B}^*\rightarrow\{0,\ldots,V-1\}^*$ 与 $D$；无损要求对支持域内所有文本满足 $D(E(s))=s$。这不要求 `encode(decode(ids)) == ids` 对任意非法 id 序列成立，因为不同 token 序列可能拼出同一 bytes；需明确你验证的是文本往返还是 id 规范化。

Normalization 是另一个函数 $N(s)$。若训练前使用 NFC，则模型实际看到 $E(N(s))$，只能保证解码回 $N(s)$，不能保证恢复未经保存的原始表示。NFKC 还会折叠兼容字符，影响精确拷贝、代码和标识符；绝不能在不记录策略时静默使用。

## 手算/数值例

字符串 `A中🙂` 有 3 个 Python code point。UTF-8 分别为 `41`、`E4 B8 AD`、`F0 9F 99 82`，共 8 bytes。若用纯 byte 词表，恰好是 8 个 token；若训练出的词表包含“中”的三 byte piece 与 emoji 的四 byte piece，则可能是 3 个 token。再比较 `é` 与 `e\u0301`：它们显示近似相同，但前者 UTF-8 为 `C3 A9`（2 bytes），后者为 `65 CC 81`（3 bytes）；NFC 后才会一致。因此“字符数=token 数”在英文、中文和 emoji 上都不是通用规则。

## 最小可运行代码

下面实现一个可逆 byte tokenizer。它没有压缩，但没有未知字符，适合作为 BPE 的正确性底座。

```python
import unicodedata
import torch

def encode_bytes(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

def decode_bytes(ids: torch.Tensor) -> str:
    values = ids.tolist()
    if any(not 0 <= i < 256 for i in values):
        raise ValueError("byte id 必须在 [0, 256)")
    return bytes(values).decode("utf-8", errors="strict")

for text in ["A中🙂", "é", "e\u0301"]:
    ids = encode_bytes(text)
    assert decode_bytes(ids) == text
    normalized = unicodedata.normalize("NFC", text)
    print(ascii(text), len(text), len(ids), ids.tolist(), ascii(normalized))
```

这里使用 `strict`，因为用替换字符吞掉非法序列会破坏无损性。生产系统可以选择错误策略，但必须把策略写入数据契约并统计异常。

## 反例/调试

不要对每个 byte token 单独调用 `bytes([i]).decode()`：中文和 emoji 的中间 byte 会报错；正确做法是先合并完整 bytes。不要用 `set(corpus)` 直接编号，集合遍历顺序和数据版本会使 id 漂移；排序并冻结映射。不要以 `len(text)` 估显存或上下文，预算对象是 token 长度。不要看到 `�` 就假设模型生成错误，它往往是过早解码了不完整 byte 前缀。Normalization 测试必须同时保留 `repr`、code point 和 bytes，肉眼显示无法发现组合序列差异。特殊 token 也不能先当普通文本 encode 再字符串替换，否则边界处可能误命中用户内容。

## 主流工作与边界

GPT-2 采用 byte-level BPE：byte 基础避免普通文本 OOV，merge 提高压缩；SentencePiece 可直接从原始 Unicode 文本训练 BPE 或 unigram，并把 normalization 和空格处理纳入模型文件。二者不是同一算法的不同名字；WordPiece 的训练目标也与 BPE 的最高频 pair 不同。现代模型仍受 tokenizer fertility、跨语言压缩率、数字和代码切分影响，但 token 越少不自动意味着模型越好：大词表增加 embedding/输出层参数，粗 piece 也可能损伤组合泛化。字节级模型消除词表 OOV，不等于消除无效 UTF-8、数据污染或 Unicode 安全问题。

## 对应 Notebook、互动图与 starter

打开 `notebooks/core/03_tokenization_and_bpe.ipynb`，先只完成 Unicode、UTF-8、字符 tokenizer 与 byte baseline 小节。使用 `docs/interactive/index.html` 的分词视图并排输入英文、中文、组合重音与 ZWJ emoji；记录字符、code point、byte、token 四个计数。本周没有必填 starter；`exercises/starter/06_byte_bpe.py` 留到第 10 周。若 notebook 输出与这里不同，优先比较 normalization 与 tokenizer 版本，而不是直接改期望值。

## 实验任务

实验 A：自建至少 20 条离线字符串集合，覆盖 ASCII、中文、阿拉伯文、换行、前后空格、组合重音、emoji 修饰符与 ZWJ；对每条执行严格往返。实验 B：分别在原文、NFC、NFKC 上统计 code point 数与 byte 数，列出发生改变的原始 `repr`，解释能否接受这种信息损失。实验 C：用字符词表与 256-byte 词表编码同一小语料，报告 OOV 数、平均 tokens/string 与词表大小；不要用测试字符串参与构词表后再宣称 OOV 为零。所有数据硬编码在脚本中，CPU/offline 可复现。

## 验收 rubric

满分 10 分：六层概念能用反例区分 2 分；UTF-8 手算与代码结果一致 1 分；严格往返覆盖复杂 Unicode 2 分；词表编号确定且 special token 无冲突 1 分；normalization 信息损失说明清楚 1 分；三种 tokenizer 指标比较公平 1 分；实验记录版本、seed 和失败串 1 分；能解释单 token 非法 UTF-8 不等于整体非法 1 分。若使用 `errors="ignore"` 仍声称无损、集合随机编号或只测 ASCII，则不通过。

## 一手来源

- [Unicode Standard Chapter 2](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-2/)：code point、编码形式与 UTF-8 的标准定义。
- [Unicode Standard Annex #15](https://www.unicode.org/reports/tr15/)：NFC/NFD/NFKC/NFKD normalization 的规范。
- [OpenAI GPT-2 官方代码仓库](https://github.com/openai/gpt-2)：byte-to-Unicode 映射、BPE encoder 与模型接口。
- [SentencePiece 原论文](https://arxiv.org/abs/1808.06226)：从原始句子训练、语言无关的 tokenizer 设计。
- [SentencePiece 官方实现](https://github.com/google/sentencepiece)：normalization、空格元符号、BPE/unigram 与确定模型文件说明。
