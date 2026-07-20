# 第 25 周：FLOPs、显存与混合精度

## 课程定位

训练失败经常不是“模型太大”这一句可以解释，而是参数、梯度、优化器状态、激活和临时 workspace 中某一项超过设备容量；速度也可能受算力、HBM 带宽、通信或数据加载限制。本周建立资源账本，并理解 FP16/BF16/FP32 在表示范围、精度和 kernel 支持上的差别。

## 学习目标

- 分项估算参数、梯度、Adam 状态、激活与通信 buffer 的字节。
- 用近似 `C≈6ND` 说明 dense Transformer 训练 FLOPs 的假设和局限。
- 解释 autocast、master weights、loss scaling 与 BF16 的作用。
- 用溢出/下溢反例定位非有限梯度，而不是只降低 batch。

## 前置

需要了解浮点指数/尾数、前向与反向、AdamW state、activation checkpointing。应能用 `tensor.numel()*tensor.element_size()` 核对估算，并区分 allocated、reserved 和峰值显存。

## 直觉

资源账本像旅行行李清单：参数只是衣服，梯度、优化器、激活和临时空间也占箱子。混合精度不是把所有张量强制变成 16 位，而是让适合低精度的矩阵乘使用硬件快路径，让归约、主权重或敏感操作保留高精度。FP16 范围较窄，loss scaling 把很小梯度暂时放大；BF16 指数范围接近 FP32，通常不需同样的 scaling，但尾数更短。

## 张量/数据契约

预算输入至少为参数量 `N`、dtype bytes、是否保留 FP32 master、梯度 dtype、optimizer state dtype、batch `B`、sequence `T`、width `D`、layers `L` 和 checkpoint 策略。输出分项字节而不是一个总数。autocast 区域只包前向和 loss；反向使用前向保存的 dtype。FP16 使用 GradScaler 时顺序为 scale loss、backward、unscale、检查/clip、step、update。日志记录 scaler、跳过 step 次数和首个非有限 tensor。

## 推导与机制

纯 FP32 AdamW 的参数、梯度、一阶矩、二阶矩约各 `4N` 字节，总主导项 `16N`；若低精度参数/梯度各 `2N`，保留 FP32 master 与两份 FP32 moments，则约 `2N+2N+4N+8N=16N`，所以混合精度不必然把 optimizer 主导内存减半。激活近似随 `B·T·D·L` 增长，具体倍数取决于保存哪些中间量。dense Transformer 常用训练 FLOPs 近似 `6ND_tokens`，忽略 embedding、attention 的 `T²` 项、重计算、稀疏激活和硬件利用率。

## 数值例

1B 参数模型用 FP32 AdamW，仅参数+梯度+两矩约 16 GB，尚未计激活和 allocator。BF16 参数/梯度、FP32 master/moments 仍约 16 GB；若框架不需独立 master，可能约 12 GB。`B=8,T=2048,D=2048,L=24` 时一个仅保存单份 hidden 的 BF16 张量链也约 `8×2048×2048×24×2≈1.5 GiB`，真实 block 会保存多份或通过 checkpoint 重算。

## 最小代码

```python
import torch
from torch import nn


def training_memory_bytes(
    params, *, param_b=2, grad_b=2, master_b=4, moment_b=4, moments=2
):
    parts = {
        "parameters": params * param_b,
        "gradients": params * grad_b,
        "master_weights": params * master_b,
        "optimizer_moments": params * moment_b * moments,
    }
    return parts, sum(parts.values())


parts, total = training_memory_bytes(1_000_000_000)
assert total == 16_000_000_000
assert sum(parts.values()) == total

# CPU 上也能验证 autocast 的边界；生产训练通常在 CUDA 上使用同一上下文契约。
layer = nn.Linear(4, 2)
tokens = torch.randn(3, 4)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    logits = layer(tokens)
assert logits.shape == (3, 2) and torch.isfinite(logits).all()
```
预算函数是数量级 baseline，autocast 示例也不是完整训练循环。生产中还要计 FSDP all-gather、NCCL buffer、CUDA graph、fused optimizer workspace、fragmentation 和框架版本；真实性能必须读取 profiler 和设备峰值。

## 反例与调试

只把模型 `.half()` 会让某些归约和 optimizer state 也低精度，容易溢出。FP16 在 `square()` 后溢出再转 FP32无法恢复，应在运算前升精度。clip 必须在 scaler unscale 后做。显存 OOM 若发生在第一步 backward 而非 forward，说明梯度/保存激活占主导；若第二步才发生，可能持有旧 graph 或未清引用。`nvidia-smi` 进程占用不是精确 tensor 分项，应配合框架 memory stats。

## 主流工作与证据等级

Mixed Precision Training 论文是 FP16 master weights/loss scaling 的基础证据；PyTorch AMP 文档是当前 API 证据。公开训练报告广泛使用 BF16、FP32 optimizer state 和 activation checkpointing，属于采用证据。不同 GPU 对 FP8/BF16 支持差异巨大，理论字节不能替代 kernel 可用性。`6ND` 是规划近似，不是逐算子 profiler 真值。

## Notebook、互动图与 starter

在 `learning/labs/07_pretraining_systems.ipynb` 比较 dtype 误差、理论/实测字节与 checkpoint 重计算；在 `learning/readings/interactive/training-and-alignment.html` 查看每设备内存分项。本周没有独立 starter 时，把资源计算器、AMP 正反例和 profiler 表作为交付物。

## 实验

对同一 Tiny GPT 跑 FP32、BF16 autocast、FP16+GradScaler，固定 batch/token/seed，记录 loss 差、step 时间、峰值内存、非有限次数。再打开 activation checkpointing，测内存和重计算时间。对 1B 假想配置列出至少两种 optimizer/master 假设，并用分项表说明差异来源。

## 验收 rubric

- 30%：资源账本分项、单位和假设完整，可由 tensor 字节核对。
- 25%：AMP 顺序、loss scaling 与 dtype 敏感操作正确。
- 25%：实验同时报告数值、内存和时间。
- 20%：明确估算、框架统计、profiler 与生产 kernel 的证据等级。

## 一手来源

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [PyTorch Automatic Mixed Precision 官方文档](https://docs.pytorch.org/docs/stable/accelerator/amp.html)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
