# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["optional", "gpu"]
# # 90｜可选 GPU：精确注意力后端观察
#
# 本 Notebook 不是必修。在 Colab/Kaggle GPU 上观察 PyTorch SDPA 后端；CPU 环境会安全跳过性能部分。FlashAttention 改执行顺序，不改变数学答案。

# %% tags=["optional", "gpu"]
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("CPU 模式：跳过 GPU 性能结论，仍可运行正确性测试。")

# %% tags=["optional", "gpu"]
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q = torch.randn(1, 4, 64, 32, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("output:", out.shape, out.device)

# %% [markdown]
# 若做性能实验，必须记录 GPU、PyTorch/CUDA、dtype、B/H/T/D、预热次数和重复次数；不同硬件的绝对速度不放进同一排名。

# %% [markdown] llm_course_enrichment=true
# ## 选修边界
#
# 核心课程全部要求 CPU/离线可运行。本 Notebook 只诊断 CUDA/MPS 与 attention 后端；没有 GPU 不是失败。不同平台的 kernel、dtype 与性能不可直接类比。

# %% llm_course_enrichment=true
import platform

import torch

print(
    {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
)

# %% [markdown] llm_course_enrichment=true
# ## 1. 统一选择设备
#
# Apple Silicon 通常用 `mps`，NVIDIA 用 `cuda`，否则回退 `cpu`。先验证正确性，再测性能；同一输入和容差比较 reference 与 fused backend。

# %% llm_course_enrichment=true
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("selected:", device)

# %% [markdown] llm_course_enrichment=true
# ## 2. SDPA 是统一接口，不保证统一后端
#
# PyTorch 依据设备、dtype、形状、mask 与版本选择 math、memory-efficient 或 FlashAttention 类 backend。运行成功不证明使用了某个 kernel。

# %% llm_course_enrichment=true
from torch.nn.functional import scaled_dot_product_attention

q = torch.randn(1, 2, 16, 8, device=device)
out = scaled_dot_product_attention(q, q, q, is_causal=True)
assert out.shape == q.shape and torch.isfinite(out).all()
print(tuple(out.shape), out.dtype)

# %% [markdown] llm_course_enrichment=true
# ## 3. 计时方法
#
# GPU 调用异步；正式计时前 warmup，并在边界同步。报告 batch、heads、T、D、dtype、mask、PyTorch/驱动与设备。

# %% llm_course_enrichment=true
import time

for _ in range(3):
    scaled_dot_product_attention(q, q, q, is_causal=True)
if device.type == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    scaled_dot_product_attention(q, q, q, is_causal=True)
if device.type == "cuda":
    torch.cuda.synchronize()
print("10 runs ms=", round((time.perf_counter() - start) * 1000, 3), "（仅烟雾测试）")

# %% [markdown] llm_course_enrichment=true
# ## 4. 内存与精度
#
# FP16、BF16、TF32 支持因设备而异；训练还受 activation、optimizer state 和 allocator 影响。不要用单点显存读数替代峰值 profiler。

# %% llm_course_enrichment=true
if device.type == "cuda":
    props = torch.cuda.get_device_properties(device)
    print(
        {
            "name": props.name,
            "memory_GiB": round(props.total_memory / 2**30, 2),
            "bf16": torch.cuda.is_bf16_supported(),
        }
    )
else:
    print("当前无 CUDA：CPU/MPS 路径仍可完成检查。")

# %% [markdown] llm_course_enrichment=true
# ## 完成标准与来源
#
# - [ ] 正确识别 device；[ ] SDPA 通过；[ ] 计时含 warmup/同步；[ ] 记录完整形状和版本；[ ] 不把 backend 推断当事实。
# - 来源：[PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MPS](https://pytorch.org/docs/stable/notes/mps.html)。
