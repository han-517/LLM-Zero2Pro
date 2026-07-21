# 大作业 03 系统实验报告

## 环境与复现

- OS、Python、PyTorch、CUDA、Triton、GPU/CPU 型号：
- 完整命令、commit、seed 与 profiler trace 路径：

## Profile

列出 top operators、CPU/CUDA self time、形状、预热/测量次数。解释同步位置；不要把异步 launch 时间当 kernel 时间。

## 正确性—吞吐—内存

| 实现 | dtype | shape | max error | median ms | tokens/s | peak memory |
|---|---|---|---:|---:|---:|---:|
| PyTorch baseline | | | | | | |
| Triton | | | | | | |
| DDP | | | | | | |
| FSDP/ZeRO-style | | | | | | |

性能结论必须报告预热、同步、分位数和至少 3 个尺寸；不设置跨机器绝对速度门槛。

## 分布式语义

- DDP 运行后的跨 rank 参数 checksum：
- bucket 划分、理论通信字节与重叠机会：
- optimizer-state shard 负载和最差/平均比：
- checkpoint 由哪些 rank 写入，如何恢复 world-size 变化：
