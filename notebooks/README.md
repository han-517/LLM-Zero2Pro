# Notebook 使用指南

## 唯一入口

在仓库根目录运行：

```text
uv run llm-course lab
```

该命令固定工作目录并打开 [`00_START_HERE.ipynb`](00_START_HERE.ipynb)。保留启动终端；退出时回到终端按 `Ctrl+C`。不要从 `core/` 中随机挑一本开始，欢迎页会先检查 Kernel、解释目录并给出依赖顺序。

## 目录结构

```text
notebooks/
├─ 00_START_HERE.ipynb       # 环境、路线和当前周导航
├─ core/                     # 文本 LLM 必修，按 01→11 学习
│  ├─ 01_shapes_and_autograd.ipynb
│  ├─ 02_neural_language_models.ipynb
│  ├─ 03_tokenization_and_bpe.ipynb
│  ├─ 04_attention_mechanics.ipynb
│  ├─ 05_tiny_gpt.ipynb
│  ├─ 06_modern_decoder.ipynb
│  ├─ 07_pretraining_systems.ipynb
│  ├─ 08_attention_frontiers.ipynb
│  ├─ 09_moe.ipynb
│  ├─ 10_posttraining.ipynb
│  └─ 11_inference_serving.ipynb
└─ optional/                 # 不计入文本主线毕业要求
   ├─ 80_multimodal_bridge.ipynb
   └─ 90_gpu_environment_check.ipynb
```

Notebook 是实验层，不承担完整讲义。`course/` 保存周次与资产契约，`docs/` 保存推导和主流工作，`exercises/starter/` 才是需要独立填写的核心代码。

## 必修顺序

| 顺序 | Notebook | 核心实验 |
|---:|---|---|
| 1 | [`core/01_shapes_and_autograd.ipynb`](core/01_shapes_and_autograd.ipynb) | 广播反例、有限差分、分支梯度 |
| 2 | [`core/02_neural_language_models.ipynb`](core/02_neural_language_models.ipynb) | Bigram、窗口 MLP、RNN/BPTT |
| 3 | [`core/03_tokenization_and_bpe.ipynb`](core/03_tokenization_and_bpe.ipynb) | UTF-8 byte、BPE merge、序列化 |
| 4 | [`core/04_attention_mechanics.ipynb`](core/04_attention_mechanics.ipynb) | Q/K/V、组合 mask、未来扰动 |
| 5 | [`core/05_tiny_gpt.ipynb`](core/05_tiny_gpt.ipynb) | 训练、梯度范数、保存/加载、生成 |
| 6 | [`core/06_modern_decoder.ipynb`](core/06_modern_decoder.ipynb) | RMSNorm、SwiGLU、RoPE、GQA、KV Cache |
| 7 | [`core/07_pretraining_systems.ipynb`](core/07_pretraining_systems.ipynb) | 数据治理、packing、AdamW、内存账本 |
| 8 | [`core/08_attention_frontiers.ipynb`](core/08_attention_frontiers.ipynb) | Flash、稀疏、MLA、线性状态 |
| 9 | [`core/09_moe.ipynb`](core/09_moe.ipynb) | Top-k、容量、dropping、router 梯度 |
| 10 | [`core/10_posttraining.ipynb`](core/10_posttraining.ipynb) | SFT shift、LoRA、DPO、GRPO 边界 |
| 11 | [`core/11_inference_serving.ipynb`](core/11_inference_serving.ipynb) | 量化、分页、推测解码、TTFT/TPOT |

选修：[`optional/80_multimodal_bridge.ipynb`](optional/80_multimodal_bridge.ipynb) 只讲视觉 token 接口；[`optional/90_gpu_environment_check.ipynb`](optional/90_gpu_environment_check.ipynb) 只做 CUDA/MPS 诊断。两者都不是核心路线前置条件。

## 每一本现在包含什么

每本必修实验至少包含 12 个单元，并统一形成：

1. 学习目标、前置知识与张量形状账本。
2. 先预测再运行的最小实验。
3. 一个会失败或推翻错误直觉的反例。
4. 经典方法与现代方法的适用边界。
5. 对应 starter ID 和核查命令。
6. 完成断言、常见误区与一手来源。

`metadata.llm_course` 记录周次、预计时间、前置知识、starter ID 和完成断言。维护者运行 `uv run python scripts/sync_notebooks.py` 只同步课程契约，不应覆盖学习者的观察单元。

## 正确的实验方式

1. 先读目标，在 Markdown 中写下预期形状和结果。
2. 逐格运行，每次只改变一个变量。
3. 报错时先读 traceback 最后一行，再检查 shape/dtype/device/数值范围。
4. 主动保留一个失败输入，写成可证伪的结论。
5. 执行 **Kernel → Restart Kernel and Run All Cells**，排除隐藏状态。
6. 在 starter 中独立实现核心代码，然后运行：

```text
uv run llm-course exercises list
uv run llm-course exercises check <编号或别名>
```

参考实现位于 `src/llm_from_scratch/`，应在公开核查通过后再比较。Starter 初始出现 `NotImplementedError` 是预期状态。

## 常见问题

- **找不到包**：关闭 Lab，在含 `pyproject.toml` 的根目录运行 `uv sync`，再用课程命令启动。
- **Kernel 不对**：关闭所有 Lab 服务后重启，不要在 Notebook 内临时 `pip install`。
- **链接跳转失败**：确认当前文件位于新的 `core/` 或 `optional/` 目录，不要使用旧书签。
- **HTML 只显示源码**：互动页应在系统浏览器打开 [`../docs/interactive/index.html`](../docs/interactive/index.html)。
- **GPU 不可用**：核心 Notebook 以 CPU 为验收基线，无需修复成 CUDA 才能继续。
- **保存输出过大**：清理模型权重、大矩阵与长日志；仓库只保留可复现实验代码。
