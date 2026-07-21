# 大作业 03：GPU profiling、Triton 与分片训练

本项目把第 25、26、29 课变成真实系统实验。CPU 学员必须完成 profiler、严谨 benchmark、通信 bucket、张量/优化器状态分片和双进程 Gloo DDP；Linux/NVIDIA 环境再完成 Triton、NCCL 与 FSDP。性能结论必须先过数值正确性，且不设置跨机器绝对速度门槛。

## 前置课次

完成第 25、26、29 课和大作业 01。托管 GPU 额外环境见 `setup/environment.md` 与 `requirements-hosted-gpu.txt`。

## CPU 与多进程必修

1. 在 `student_systems/systems.py` 完成 warmup/sync/quantile benchmark 与 PyTorch Profiler 聚合。
2. 实现不均匀 contiguous shard/gather、按字节 communication bucket 和 optimizer-state 负载分配。
3. 实现可同时接受普通模型与 `DistributedDataParallel` 的 next-token 训练步。
4. 运行：

```text
uv run llm-course projects check 03
uv run python learning/labs/projects/03_gpu_systems/run_profile.py --device cpu
uv run torchrun --standalone --nproc-per-node=2 learning/labs/projects/03_gpu_systems/run_ddp.py
```

最后一条必须真实启动两个进程；脚本检查不同本地 batch 更新后各 rank 参数 checksum 完全一致。

## Linux/NVIDIA 扩展

在托管 GPU 环境安装附加依赖后：

```text
uv pip install -r requirements-hosted-gpu.txt
uv run python learning/labs/projects/03_gpu_systems/run_profile.py --device cuda
uv run torchrun --standalone --nproc-per-node=2 learning/labs/projects/03_gpu_systems/run_ddp.py
uv run torchrun --standalone --nproc-per-node=2 learning/labs/projects/03_gpu_systems/run_fsdp.py
RUN_GPU_PROJECT_TESTS=1 uv run pytest -q checks/projects/test_03_triton_optional.py
```

Windows PowerShell 设置环境变量使用 `$env:RUN_GPU_PROJECT_TESTS='1'`，但 Triton 任务本身要求 Linux/NVIDIA。`triton_matmul.py` 先实现非整 tile 形状的 FP32 accumulation forward；随后把同样的 online-softmax 分块思想迁移到第 29 课 attention starter。课程不会把一个 matmul kernel 宣称为完整 FlashAttention。

## 完成标准

- CPU 公开核查通过；profile JSON 包含真实算子 self time，benchmark 记录预热、同步、中位数与 p20/p80。
- `torchrun` 双进程 smoke 通过；报告 bucket 字节、理论通信量和参数同步证据。
- optimizer-state 分配给出每 rank 负载和最差/平均比；说明它与真实 ZeRO/FSDP flatten/shard 的差异。
- GPU 扩展对至少 3 个矩阵形状比较 PyTorch/Triton 的最大误差、median latency 与峰值显存；保留慢于 baseline 的结果。
- FSDP 写出每 rank shard，并说明 world-size 改变时为何需要合并或重分片 checkpoint。

FSDP API 可能随 PyTorch 版本演化；以锁定环境中的官方 API 为准，并在报告中记录版本。本项目不要求购买本地 GPU。