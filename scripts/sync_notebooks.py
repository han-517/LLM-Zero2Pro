"""从分层课程清单同步 Notebook 契约，并创建可离线执行的阶段实验。"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from llm_course.course import load_roadmap

ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "notebooks/00_START_HERE.ipynb": (20, ["会打开终端"], "能运行 doctor 并找到当前周资产"),
    "notebooks/core/01_shapes_and_autograd.ipynb": (
        100,
        ["Python 基础"],
        "有限差分与分支反传断言通过",
    ),
    "notebooks/core/02_neural_language_models.ipynb": (
        120,
        ["张量形状", "交叉熵"],
        "Bigram/MLP/RNN 的损失与因果性断言通过",
    ),
    "notebooks/core/03_tokenization_and_bpe.ipynb": (
        90,
        ["Unicode 与 Python bytes"],
        "Byte BPE encode/decode 往返一致",
    ),
    "notebooks/core/04_attention_mechanics.ipynb": (
        100,
        ["矩阵乘法", "Softmax"],
        "未来扰动和全遮蔽行断言通过",
    ),
    "notebooks/core/05_tiny_gpt.ipynb": (130, ["注意力", "残差连接"], "TinyGPT 可保存、加载并生成"),
    "notebooks/core/06_modern_decoder.ipynb": (
        120,
        ["经典 Decoder"],
        "经典/现代配置的形状与 cache 等价断言通过",
    ),
    "notebooks/core/07_pretraining_systems.ipynb": (
        110,
        ["训练循环"],
        "去重、packing、AdamW 与内存账本断言通过",
    ),
    "notebooks/core/08_attention_frontiers.ipynb": (
        120,
        ["RoPE", "KV Cache"],
        "稀疏 mask 与线性注意力两条路径数值一致",
    ),
    "notebooks/core/09_moe.ipynb": (110, ["现代 Decoder"], "路由、容量、负载和梯度断言通过"),
    "notebooks/core/10_posttraining.ipynb": (
        120,
        ["next-token loss"],
        "response mask、LoRA、DPO/GRPO toy 断言通过",
    ),
    "notebooks/core/11_inference_serving.ipynb": (
        120,
        ["KV Cache", "概率采样"],
        "分页、推测采样与 TTFT/TPOT 指标断言通过",
    ),
}

NEW_NOTEBOOK_CELLS = {
    "notebooks/core/03_tokenization_and_bpe.ipynb": [
        (
            "markdown",
            """# Byte BPE：字符、字节与 token

先观察同一字符串的字符数与 UTF-8 字节数，再训练确定性的教学版 Byte BPE。""",
        ),
        (
            "code",
            """from llm_from_scratch.tokenization import BytePairTokenizer

text = "LLM 学习：hello hello"
raw = list(text.encode("utf-8"))
tokenizer = BytePairTokenizer.train(text * 4, vocab_size=264)
ids = tokenizer.encode(text)
print({"characters": len(text), "bytes": len(raw), "tokens": len(ids)})
assert tokenizer.decode(ids) == text""",
        ),
        (
            "markdown",
            """## 练习

打开 exercises/starter/06_byte_bpe.py。核心 merge 逻辑保持空缺；完成后运行：

    uv run llm-course exercises check 06""",
        ),
    ],
    "notebooks/core/06_modern_decoder.ipynb": [
        (
            "markdown",
            """# 经典 Decoder → 现代 Decoder

一次只改变一个组件：LayerNorm/RMSNorm、GELU/SwiGLU、绝对位置/RoPE、MHA/GQA。""",
        ),
        (
            "code",
            """import torch
from llm_from_scratch.transformer import GPTConfig, TinyGPT

common = dict(block_size=16, n_layer=1, n_head=4, d_model=32, dropout=0.0)
classic = TinyGPT(GPTConfig.classic(64, **common))
modern = TinyGPT(GPTConfig.modern(64, **common))
tokens = torch.randint(0, 64, (2, 6))
classic_logits, _, _ = classic(tokens, return_caches=False)
modern_logits, _, modern_cache = modern(tokens)
assert classic_logits.shape == modern_logits.shape == (2, 6, 64)
assert modern.position_embedding is None
assert modern_cache is not None
print({"classic_parameters": sum(p.numel() for p in classic.parameters()),
       "modern_parameters": sum(p.numel() for p in modern.parameters())})""",
        ),
        (
            "markdown",
            "RoPE 位于 attention 内部，只旋转 Q/K；cache 保存旋转后的 K。"
            "完成 07、08、03 后分别核查。",
        ),
    ],
    "notebooks/core/07_pretraining_systems.ipynb": [
        (
            "markdown",
            """# 预训练数据与训练系统

玩具实验不替代真实数据治理；它强制记录去重、边界、污染和内存假设。""",
        ),
        (
            "code",
            """import torch
from llm_from_scratch.training import (
    exact_deduplicate, pack_documents, training_memory_ledger, warmup_cosine_lr,
)

docs, duplicates = exact_deduplicate(["alpha", "beta", "alpha"])
packed, mask = pack_documents([[1, 2], [3]], block_size=4, eos_token_id=9, pad_token_id=0)
ledger = training_memory_ledger(1_000, world_size=2, shard_optimizer=True)
curve = [warmup_cosine_lr(i, total_steps=8, warmup_steps=2, max_lr=1e-3) for i in range(9)]
assert docs == ["alpha", "beta"] and duplicates == [2]
assert packed.shape == mask.shape and ledger["total"] > 0
assert curve[0] == 0 and curve[2] == max(curve)
print({"packed": packed.tolist(), "loss_mask": mask.tolist(), "ledger": ledger})""",
        ),
        (
            "markdown",
            """下一步完成 14、15；报告每条规则的保留率，不把 toy 内存账本称为峰值显存保证。""",
        ),
    ],
    "notebooks/core/08_attention_frontiers.ipynb": [
        (
            "markdown",
            """# 注意力前沿：IO、稀疏、latent cache 与线性状态

本实验关注数学契约和成本边界，不声称复现生产 kernel。""",
        ),
        (
            "code",
            """import torch
from llm_from_scratch.attention import (
    causal_linear_attention,
    causal_linear_attention_parallel,
    mla_cache_cost,
    sliding_window_mask,
)

mask = sliding_window_mask(2, 2, key_length=4)
q = torch.randn(1, 2, 6, 4)
k = torch.randn(1, 2, 6, 4)
v = torch.randn(1, 2, 6, 3)
recurrent = causal_linear_attention(q, k, v)
parallel = causal_linear_attention_parallel(q, k, v)
torch.testing.assert_close(recurrent, parallel, atol=2e-5, rtol=2e-5)
cost = mla_cache_cost(batch_size=1, layers=1, sequence_length=128, d_model=32, latent_dim=16)
print(mask.int(), cost)""",
        ),
        (
            "markdown",
            "MLA 部分是 latent-cache reconstruction baseline；需要区分缓存压缩、"
            "历史重建与 absorbed decode。",
        ),
    ],
    "notebooks/core/10_posttraining.ipynb": [
        (
            "markdown",
            """# 后训练：mask、log-prob 与相对目标

先验证 token 对齐，再看 SFT、DPO 和组内优势；PPO/GRPO 函数均是 toy objective。""",
        ),
        (
            "code",
            """import torch
from llm_from_scratch.post_training import group_relative_advantages, sequence_logprob

logits = torch.log_softmax(torch.randn(2, 4, 7), dim=-1)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
scores = sequence_logprob(logits, labels, mask)
advantages = group_relative_advantages(torch.tensor([[2.0, 2.0], [1.0, 3.0]]))
assert scores.shape == (2,)
torch.testing.assert_close(advantages[0], torch.zeros(2))
print({"sequence_logprob": scores, "advantages": advantages})""",
        ),
        (
            "markdown",
            "完成 04 与 19。任何真实结论都需额外报告 reference policy、"
            "response mask、KL、奖励和安全评测。",
        ),
    ],
    "notebooks/core/11_inference_serving.ipynb": [
        (
            "markdown",
            """# 推理服务：分页缓存、推测解码与服务指标

PageTable 是内存管理模拟器；它不存真实 K/V，也不是 PagedAttention kernel。""",
        ),
        (
            "code",
            """from llm_from_scratch.inference import PageTable, RequestTrace, summarize_serving

table = PageTable(page_size=4, free_pages=list(range(8)))
table.append_tokens("a", 5)
table.share_prefix("a", "b")
table.append_tokens("b", 1)
assert table.sequence_pages["a"][-1] != table.sequence_pages["b"][-1]
traces = [
    RequestTrace(0.0, 0.2, 1.0, output_tokens=5, prompt_tokens=8),
    RequestTrace(0.1, 0.5, 1.4, output_tokens=4, prompt_tokens=3),
]
metrics = summarize_serving(traces, ttft_slo=0.5, tpot_slo=0.3)
print({"fragmentation": table.internal_fragmentation_tokens,
       "ttft_p95": metrics.ttft.p95,
       "token_throughput": metrics.output_token_throughput})""",
        ),
        (
            "markdown",
            "完成 20，并区分 greedy 教学基线与具有接受率、残差分布和 bonus token 的随机正确版本。",
        ),
    ],
}


def build_missing(path: Path, relative_path: str) -> None:
    cells = NEW_NOTEBOOK_CELLS.get(relative_path)
    if cells is None:
        raise FileNotFoundError(f"roadmap 引用了缺失且无生成规格的 Notebook: {relative_path}")
    notebook_cells = [
        new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
        for kind, source in cells
    ]
    notebook = new_notebook(cells=notebook_cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


def main() -> None:
    data = load_roadmap()
    weeks_by_notebook: dict[str, set[int]] = {}
    starters_by_notebook: dict[str, set[str]] = {}
    for raw_week, asset in data["assets"].items():
        week = int(raw_week)
        for relative_path in asset["notebooks"]:
            weeks_by_notebook.setdefault(relative_path, set()).add(week)
            starters_by_notebook.setdefault(relative_path, set()).update(
                map(str, asset["exercises"])
            )

    for relative_path, weeks in weeks_by_notebook.items():
        path = ROOT / relative_path
        if not path.exists():
            build_missing(path, relative_path)
        notebook = nbformat.read(path, as_version=4)
        minutes, prerequisites, assertion = SPECS[relative_path]
        contract = {
            "weeks": sorted(weeks),
            "estimated_minutes": minutes,
            "prerequisites": prerequisites,
            "starter_ids": sorted(starters_by_notebook[relative_path]),
            "completion_assertion": assertion,
            "offline_cpu": True,
        }
        notebook.metadata["llm_course"] = contract
        preamble = (
            f"**课程契约** · 周次 {', '.join(map(str, sorted(weeks)))} · "
            f"预计 {minutes} 分钟 · Starter "
            f"{', '.join(contract['starter_ids']) or '研究产出'} · 默认 CPU/离线。"
        )
        marked = [cell for cell in notebook.cells if cell.metadata.get("llm_course_preamble")]
        if marked:
            marked[0].source = preamble
        else:
            cell = new_markdown_cell(preamble)
            cell.metadata["llm_course_preamble"] = True
            notebook.cells.insert(0, cell)
        nbformat.write(notebook, path)

    optional_path = ROOT / "notebooks" / "optional" / "80_multimodal_bridge.ipynb"
    if not optional_path.exists():
        optional = new_notebook(
            cells=[
                new_markdown_cell(
                    "# 多模态选修：patchify → vision embedding → projector → 文本 Decoder 接口"
                ),
                new_code_cell(
                    "import torch\n"
                    "images = torch.arange(2*3*4*4.).reshape(2,3,4,4)\n"
                    "patches = images.unfold(2,2,2).unfold(3,2,2)\n"
                    "patches = patches.permute(0,2,3,1,4,5).reshape(2,4,12)\n"
                    "projector = torch.nn.Linear(12, 16)\n"
                    "vision_tokens = projector(patches)\n"
                    "assert vision_tokens.shape == (2,4,16)\n"
                    "print(vision_tokens.shape)"
                ),
                new_markdown_cell(
                    "不下载权重、不训练大型 VLM。完成 "
                    "exercises/starter/21_multimodal_bridge.py 后运行独立核查。"
                ),
            ],
            metadata={
                "llm_course": {
                    "weeks": [],
                    "estimated_minutes": 60,
                    "prerequisites": ["TinyGPT", "线性投影"],
                    "starter_ids": ["21"],
                    "completion_assertion": "视觉 token 的形状可接入文本 Decoder d_model",
                    "offline_cpu": True,
                    "optional": True,
                }
            },
        )
        nbformat.write(optional, optional_path)


if __name__ == "__main__":
    main()
