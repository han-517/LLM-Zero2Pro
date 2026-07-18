# Notebook 使用顺序

1. `00_shapes_and_autograd.ipynb`：形状和链式法则。
2. `01_attention_lab.ipynb`：因果注意力和权重图。
3. `02_moe_lab.ipynb`：路由负载、容量和 dropping。
4. `03_tiny_gpt.ipynb`：单 batch 过拟合闭环。
5. `90_optional_gpu_check.ipynb`：免费 GPU 可选，不计入必修验收。

从仓库根目录运行 `uv run jupyter lab`。必修 Notebook 会在 CPU 上由测试自动执行。

无需启动 Jupyter 的基础交互实验见 `docs/interactive/core-concepts.html`；它覆盖 Softmax、
因果注意力、KV Cache 与 MoE。完成 Notebook 01/02 后，打开
`docs/interactive/architecture-evolution.html` 对齐 RoPE、注意力和 MoE 的演化与当前公开模型。

