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

# %% [markdown] llm_course_preamble=true
# **课程契约** · 周次 14, 15 · 预计 130 分钟 · Starter 13 · 默认 CPU/离线。

# %% [markdown]
# # 03｜Tiny GPT 单 batch 过拟合
#
# 目的不是训练好模型，而是证明数据、因果注意力、loss、反传和更新形成闭环。

# %%
import torch

from llm_from_scratch.transformer import GPTConfig, TinyGPT

torch.manual_seed(12)
model = TinyGPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=2, d_model=8))
x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
y = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
losses = []
for _step in range(35):
    _, loss, _ = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print("initial:", round(losses[0], 4), "final:", round(losses[-1], 4))
assert losses[-1] < losses[0] * 0.35

# %%
print("loss samples:", [round(value, 4) for value in losses[:: max(1, len(losses) // 6)]])

# %% [markdown]
# 过拟合成功仍不能证明 mask、target shift 和验证切分正确；这些由独立测试负责。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：把零件连成 Decoder-only LM
#
# `token ids → position representation → N×(pre-norm attention + residual + pre-norm MLP + residual) → final norm → lm head`。训练目标右移一位；残差流保持 `[B,T,d_model]`。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.transformer import GPTConfig, TinyGPT

config = GPTConfig.classic(
    vocab_size=32, block_size=8, n_layer=1, n_head=2, d_model=16, dropout=0.0
)
model = TinyGPT(config)
x = torch.randint(0, 32, (2, 6))
y = torch.randint(0, 32, (2, 6))
logits, loss, caches = model(x, y, return_caches=False)
assert logits.shape == (2, 6, 32) and loss.ndim == 0 and caches is None
print("parameters=", model.parameter_count(), "loss=", float(loss))

# %% [markdown] llm_course_enrichment=true
# ## 1. 单 batch 过拟合是连线测试
#
# 它能检查 shift、梯度、优化器和容量是否基本连通，但不能证明泛化。除总 loss 外还要记录各层梯度范数；某层长期为零通常表示图被截断、mask 错误或参数未进 optimizer。

# %% llm_course_enrichment=true
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
optimizer.zero_grad()
_, loss, _ = model(x, y, return_caches=False)
loss.backward()
grad_norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
assert grad_norms and torch.isfinite(torch.tensor(list(grad_norms.values()))).all()
optimizer.step()
print(dict(list(grad_norms.items())[:4]))

# %% [markdown] llm_course_enrichment=true
# ## 2. 保存、加载与生成属于同一验收
#
# 只保存 `state_dict` 不够：还要保存配置和 tokenizer 契约。这里用临时目录避免污染仓库，并用温度 0 做确定性生成。

# %% llm_course_enrichment=true
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "tiny.pt"
    torch.save({"config": config, "model": model.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    restored = TinyGPT(payload["config"])
    restored.load_state_dict(payload["model"])
    generated = restored.generate(x[:1, :3], max_new_tokens=2, temperature=0)
assert generated.shape == (1, 5)
print("generated ids=", generated.tolist())

# %% [markdown] llm_course_enrichment=true
# ## 3. 常见失败
#
# Targets 未右移会学成复制器；训练时无谓构造 cache 会增加内存；`eval()` 后需恢复模式；生成超过 block size 必须定义滑窗/位置重编号。

# %% llm_course_enrichment=true
try:
    GPTConfig(vocab_size=32, n_head=0, d_model=16)
except ValueError as error:
    print("配置校验生效：", error)

# %% [markdown] llm_course_enrichment=true
# ## 练习与验收
#
# 填写 starter 09、10。完成标准：loss 明显下降、梯度有限、checkpoint 往返一致、能生成且训练默认不创建 cache。

# %% [markdown] llm_course_enrichment=true
# ## 来源与边界
#
# [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)、[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。这是 CPU 教学模型，不复现生产吞吐、数据规模或分布式容错。
