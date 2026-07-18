# Notebook 使用指南

## 唯一推荐入口

在仓库根目录运行：

```text
uv run llm-course lab
```

它会固定工作目录并直接打开 `notebooks/00_START_HERE.ipynb`。保留启动终端；关闭 JupyterLab 时回到终端按 `Ctrl+C`，确认停止服务。若浏览器未自动打开，复制终端打印的完整 localhost 地址。

## 第一次进入 Lab

1. 只打开 `00_START_HERE.ipynb`，运行环境检查单元。
2. 确认 Python 来自仓库 `.venv`，项目根目录和 PyTorch 检查均通过。
3. 运行 `uv run llm-course course path --weeks 15`，不确定时选择 15 周路线。
4. 打开当前周 Notebook；先写形状和结果预测，再运行代码。
5. 完成后执行 **Kernel → Restart Kernel and Run All Cells**。
6. 保存 Notebook，回到终端核查对应 starter。

## Notebook 地图

| 顺序 | 文件 | 覆盖主题 |
|---:|---|---|
| 0 | `00_START_HERE.ipynb` | 环境、路线、目录和核查入口 |
| 1 | `00_shapes_and_autograd.ipynb` | 广播、计算图、有限差分、分支求和 |
| 2 | `neural_lm_lab.ipynb` | 数据窗口、Bigram、MLP、RNN/BPTT |
| 3 | `tokenization_lab.ipynb` | Unicode、字节、教学版 Byte BPE |
| 4 | `01_attention_lab.ipynb` | Q/K/V、缩放、causal/padding mask |
| 5 | `03_tiny_gpt.ipynb` | 训练、梯度范数、保存/加载、生成 |
| 6 | `modern_decoder_lab.ipynb` | RMSNorm、SwiGLU、RoPE、GQA、KV Cache |
| 7 | `pretraining_lab.ipynb` | 数据治理、packing、AdamW、调度与评测 |
| 8 | `attention_frontiers_lab.ipynb` | Flash/稀疏/MLA/线性注意力/Delta 状态 |
| 9 | `02_moe_lab.ipynb` | Top-k、容量、dropping、路由梯度与负载 |
| 10 | `posttraining_lab.ipynb` | SFT、LoRA、DPO、GRPO/RLVR 边界 |
| 11 | `inference_serving_lab.ipynb` | 量化、分页缓存、连续批处理、推测解码 |
| 选修 | `80_multimodal_bridge.ipynb` | patchify → vision embedding → projector → 文本接口 |
| 选修 | `90_optional_gpu_check.ipynb` | 托管 Linux GPU 诊断 |

每个必修 Notebook 都有 `metadata.llm_course`：周次、预计时间、前置知识、starter ID 和完成断言。它们来自 `course/roadmap.yaml`，维护者可运行 `uv run python scripts/sync_notebooks.py` 同步。

## 正确的实验方式

1. 先读学习目标，不要第一次就 Run All。
2. 在 Markdown 单元写下张量形状、预期结果和一个失败反例。
3. 逐格运行；报错先看 traceback 最后一行。
4. 每次只改一个变量，记录是否支持原假设。
5. 把结论写成可证伪句子。
6. 重启 Kernel 并全部运行，排除隐藏状态。
7. 在 starter 独立实现核心代码；Notebook 图形不能替代数值核查。

## 与模板配合

```text
uv run llm-course exercises list
uv run llm-course exercises check 11
uv run llm-course exercises check autograd
```

starter 初始包含 `NotImplementedError` 是预期状态。参考实现位于 `src/llm_from_scratch/`，应在自己的公开核查通过后再比较。

## 无需 Kernel 的互动图

从 [互动实验总入口](../docs/interactive/index.html)进入基础、架构、训练对齐、推理服务与多模态选修页面。页面离线运行；如果 JupyterLab 只显示 HTML 源码，请用系统浏览器打开文件。

## 常见问题

- **找不到 `llm_course`**：关闭服务，回到含 `pyproject.toml` 的根目录，运行 `uv sync` 和 `uv run llm-course lab`。
- **Kernel 不是 Python 3.12**：不要在 Lab 内临时安装 Python；关闭所有服务后用课程命令重启。
- **端口冲突**：运行 `uv run llm-course lab --port 8890`。
- **远程机器**：运行 `uv run llm-course lab --no-browser`，按 SSH 环境安全转发端口。
- **保存失败**：确认仓库目录可写且磁盘有空间；不要把模型权重或大数据嵌进输出。
