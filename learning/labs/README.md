# 实验目录

本目录只放需要运行、修改或填写的学习资产。完整顺序由 [`learning/README.md`](../README.md) 决定，不要根据文件编号跳课。

## 两种实验格式

每本实验有两种等价表示：

- `.ipynb`：适合 VS Code Notebook 或 JupyterLab。
- `.py`：Jupytext `py:percent`，使用 `# %%` 单元，适合 VS Code Interactive Window。

同一实验任选一种。`.ipynb` 是课程维护源，`.py` 由 `uv run python scripts/sync_labs.py` 生成；个人学习记录请集中在你选择的那个文件中。

## 实验主题

| 编号 | `.ipynb` / `.py` | 服务课次 | 核心内容 |
|---:|---|---:|---|
| 01 | `01_shapes_and_autograd` | 2–4 | 广播、有限差分、分支梯度、Softmax |
| 02 | `02_neural_language_models` | 5–8 | Bigram、MLP、RNN/BPTT |
| 03 | `03_tokenization_and_bpe` | 9–10 | UTF-8、Byte BPE、序列化 |
| 04 | `04_attention_mechanics` | 11–13 | Q/K/V、缩放、组合 mask、未来扰动 |
| 05 | `05_tiny_gpt` | 14–15 | Block、训练、保存/加载、生成 |
| 06 | `06_modern_decoder` | 16–21 | RMSNorm、SwiGLU、RoPE、GQA、KV Cache |
| 07 | `07_pretraining_systems` | 22–28 | 数据治理、packing、AdamW、资源与评测 |
| 08 | `08_attention_frontiers` | 29–34 | Flash、稀疏、MLA、线性/Delta 状态 |
| 09 | `09_moe` | 35–39 | Top-k、容量、dropping、router 梯度 |
| 10 | `10_posttraining` | 40–44 | SFT、LoRA、奖励、DPO、GRPO |
| 11 | `11_inference_serving` | 45–48 | 量化、分页、连续批处理、推测解码 |

`optional/` 中是多模态桥接和 GPU 环境实验，不计入文本主线毕业要求。

## 正确运行方式

1. 从课程目录进入当前课，只运行该课列出的实验部分。
2. 先写下预期形状和结果，再运行单元。
3. 每次只改变一个变量，并保留至少一个失败反例。
4. `.ipynb` 完成后执行 **Restart Kernel and Run All Cells**；`.py` 则从干净终端完整运行。
5. 到 `starter/` 填写核心代码，再运行课程目录给出的核查命令。

参考实现位于 `src/llm_from_scratch/`，只应在 starter 通过后比较。