# 第 26 周：并行训练、检查点与故障恢复

## 课程定位

当单卡无法容纳模型或吞吐不足时，需要在复制、分片和流水线之间选择。本周不要求个人电脑启动多机集群，而是建立通信与状态所有权模型，并设计能从不同故障点恢复的 checkpoint。目标是解释每个 tensor 在哪个 rank、何时 all-gather/reduce，以及恢复后数据顺序为什么仍可追溯。

## 学习目标

- 区分 DDP、ZeRO/FSDP、tensor、pipeline 与 sequence/context parallel。
- 为参数、梯度、optimizer state 和激活画所有权/通信图。
- 估算分片后的每设备内存及 all-reduce、all-gather、reduce-scatter 边界。
- 定义包含模型、优化器、scheduler、scaler、RNG 和数据游标的可恢复 checkpoint。

## 前置

需要理解 collective communication、global/local batch、Adam 状态和混合精度。应能计算 world size、data parallel degree 与 gradient accumulation 共同决定的 global batch tokens。

## 直觉

DDP 像每个工人各有完整工具箱，处理不同样本后汇总梯度；简单但工具箱重复。ZeRO/FSDP 把工具分给不同工人，需要使用某层时临时收集，再把梯度分散回去。Tensor parallel 把一把大工具拆成几段，层内频繁通信；pipeline parallel 把工序分到不同站点，会产生等待气泡。checkpoint 不只是模型照片，还必须保存工厂进度、随机状态和原料传送位置。

## 张量/数据契约

DDP 每 rank 保有完整参数与 optimizer state，梯度在 backward 中 all-reduce。ZeRO-1 分 optimizer，ZeRO-2 再分梯度，ZeRO-3/FSDP 再分参数；具体 runtime 名称以官方版本为准。checkpoint manifest 记录 world size、mesh、shard metadata、model/config/tokenizer hash、optimizer/scheduler/scaler step、Python/CPU/CUDA RNG、sampler epoch 与 offset、数据快照和代码提交。写入采用临时目录与完成标记，防止半写 checkpoint 被当作有效。

## 推导与机制

若模型状态主导字节为参数 `P`、梯度 `G`、optimizer `O`，DDP 每 rank 约 `P+G+O`；理想 ZeRO-1 为 `P+G+O/W`，ZeRO-2 为 `P+(G+O)/W`，ZeRO-3 为 `(P+G+O)/W`，另加当前 all-gather 层、激活和通信 buffer。DDP global batch tokens 为 `local_batch·T·W·accumulation`。Pipeline 利用率粗略受 microbatch 数与 stage 数影响，microbatch 太少时首尾填充/排空气泡显著。

## 数值例

假设参数 8 GB、梯度 8 GB、optimizer 24 GB，world size 8。DDP 每 rank 主导 40 GB；理想 ZeRO-1 为 `8+8+3=19 GB`，ZeRO-2 为 `8+4=12 GB`，ZeRO-3 为 5 GB。真实 FSDP 还会 all-gather 当前参数、预取下一层并受 wrap 粒度影响，不能拿 5 GB 当峰值承诺。local batch 2、T=2048、8 rank、累积 4 时 global batch 为 131072 tokens/update。

## 最小代码

```python
import torch
from torch import nn


def global_batch_tokens(local_batch, seq_len, world_size, grad_accum):
    return local_batch * seq_len * world_size * grad_accum


model = nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": 0,
    "rng_cpu": torch.get_rng_state(),
    "data_cursor": {"epoch": 0, "sample_offset": 0},
    "config_hash": "toy-config-v1",
}
required = {"model", "optimizer", "scheduler", "step", "rng_cpu", "data_cursor"}
assert required <= checkpoint.keys()
assert global_batch_tokens(2, 2048, 8, 4) == 131_072
```
这是单进程完整 state baseline。生产分片 checkpoint 不能简单让所有 rank 写同一文件；需使用官方 distributed checkpoint API、shard metadata、原子提交和校验。不同 world size 恢复还涉及 reshard。

## 反例与调试

DDP 中每个 rank 读取同一批样本会把有效 batch 退化，检查 sampler rank/seed。梯度累积使用 `no_sync` 时最后一个 microstep 必须同步。FSDP wrap 太细会产生大量小 all-gather，太粗会提高峰值内存。checkpoint 只存模型导致恢复后 Adam 矩丢失、LR 重启，loss 会跳。保存了 RNG 却没有数据 cursor，也无法重现下一批。多 rank 写完成标记前必须确认所有 shard 成功。

## 主流工作与证据等级

ZeRO 论文提供分片状态与通信分析；Megatron-LM 论文/官方代码展示 tensor/pipeline 并行；PyTorch FSDP 官方文档是当前 API 与限制真值。公开模型报告中的 GPU 数和吞吐属于特定网络、kernel、batch 与故障策略证据，不能直接移植。课程纸面模拟验证所有权，不等于真实 NCCL 性能。

## Notebook、互动图与 starter

在 `docs/interactive/training-and-alignment.html` 切换 DDP/ZeRO/FSDP 并观察主导内存；在 `notebooks/core/07_pretraining_systems.ipynb` 绘制拓扑、估算通信并做单进程 checkpoint round-trip。本周无多机 starter，交付并行方案和可恢复 manifest。

## 实验

先用纸面 1B/7B 配置比较 DDP、ZeRO 三阶段和 tensor parallel 的内存/通信。单机用两进程 DDP（若环境允许）验证分片 sampler；否则用 mock rank 生成不重叠索引。训练 20 step，在第 11 步中断恢复，要求第 12 步 batch ID、LR、loss 和参数与不中断 run 一致。

## 验收 rubric

- 30%：并行拓扑、tensor 所有权和 collective 边界正确。
- 25%：内存估算包含临时 all-gather/激活边界说明。
- 30%：checkpoint 中断恢复达到下一步一致并验证 shard 完整。
- 15%：明确纸面/单机实验与生产网络、官方 API 边界。

## 一手来源

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [PyTorch FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Distributed Checkpoint 官方文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)
