"""同步课程实验契约，并生成 Jupytext py:percent 镜像。"""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext
import nbformat

from llm_course.course import load_roadmap

ROOT = Path(__file__).resolve().parents[1]

LAB_SPECS = {
    "learning/labs/01_shapes_and_autograd.ipynb": (
        100,
        ["Python 基础"],
        "有限差分与分支反传断言通过",
    ),
    "learning/labs/02_neural_language_models.ipynb": (
        120,
        ["张量形状", "交叉熵"],
        "Bigram/MLP/RNN 的损失与因果性断言通过",
    ),
    "learning/labs/03_tokenization_and_bpe.ipynb": (
        90,
        ["Unicode 与 Python bytes"],
        "Byte BPE encode/decode 往返一致",
    ),
    "learning/labs/04_attention_mechanics.ipynb": (
        100,
        ["矩阵乘法", "Softmax"],
        "未来扰动和全遮蔽行断言通过",
    ),
    "learning/labs/05_tiny_gpt.ipynb": (
        130,
        ["注意力", "残差连接"],
        "TinyGPT 可保存、加载并生成",
    ),
    "learning/labs/06_modern_decoder.ipynb": (
        120,
        ["经典 Decoder"],
        "经典/现代配置的形状与 cache 等价断言通过",
    ),
    "learning/labs/07_pretraining_systems.ipynb": (
        110,
        ["训练循环"],
        "去重、packing、AdamW 与内存账本断言通过",
    ),
    "learning/labs/08_attention_frontiers.ipynb": (
        120,
        ["RoPE", "KV Cache"],
        "稀疏 mask 与线性注意力两条路径数值一致",
    ),
    "learning/labs/09_moe.ipynb": (
        110,
        ["现代 Decoder"],
        "路由、容量、负载和梯度断言通过",
    ),
    "learning/labs/10_posttraining.ipynb": (
        120,
        ["next-token loss"],
        "response mask、LoRA、DPO/GRPO toy 断言通过",
    ),
    "learning/labs/11_inference_serving.ipynb": (
        120,
        ["KV Cache", "概率采样"],
        "分页、推测采样与 TTFT/TPOT 指标断言通过",
    ),
}

OPTIONAL_LABS = (
    "learning/labs/optional/80_multimodal_bridge.ipynb",
    "learning/labs/optional/90_gpu_environment_check.ipynb",
)


def _contracts() -> dict[str, tuple[set[int], set[str]]]:
    data = load_roadmap()
    result: dict[str, tuple[set[int], set[str]]] = {}
    for raw_week, asset in data["assets"].items():
        lab = asset.get("lab")
        if not isinstance(lab, dict):
            continue
        path = str(lab["ipynb"])
        weeks, starters = result.setdefault(path, (set(), set()))
        weeks.add(int(raw_week))
        starters.update(str(item).zfill(2) for item in asset["exercises"])
    return result


def _python_text(notebook: nbformat.NotebookNode) -> str:
    text = jupytext.writes(notebook, fmt="py:percent")
    return text if text.endswith("\n") else text + "\n"


def sync_labs(*, check: bool = False) -> int:
    contracts = _contracts()
    errors: list[str] = []
    for relative_path, (minutes, prerequisites, assertion) in LAB_SPECS.items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"缺少 Notebook: {relative_path}")
            continue
        notebook = nbformat.read(path, as_version=4)
        weeks, starters = contracts[relative_path]
        expected_contract = {
            "weeks": sorted(weeks),
            "estimated_minutes": minutes,
            "prerequisites": prerequisites,
            "starter_ids": sorted(starters),
            "completion_assertion": assertion,
            "offline_cpu": True,
        }
        if check and dict(notebook.metadata.get("llm_course", {})) != expected_contract:
            errors.append(f"Notebook 契约漂移: {relative_path}")
        notebook.metadata["llm_course"] = expected_contract
        python_path = path.with_suffix(".py")
        expected_python = _python_text(notebook)
        if check:
            try:
                actual_python = python_path.read_text(encoding="utf-8")
            except OSError:
                errors.append(f"缺少 VS Code 实验: {python_path.relative_to(ROOT)}")
            else:
                if actual_python.replace("\r\n", "\n") != expected_python.replace("\r\n", "\n"):
                    errors.append(f"双格式实验漂移: {python_path.relative_to(ROOT)}")
        else:
            nbformat.write(notebook, path)
            python_path.write_text(expected_python, encoding="utf-8")

    for relative_path in OPTIONAL_LABS:
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"缺少选修 Notebook: {relative_path}")
            continue
        notebook = nbformat.read(path, as_version=4)
        python_path = path.with_suffix(".py")
        expected_python = _python_text(notebook)
        if check:
            if not python_path.is_file() or python_path.read_text(encoding="utf-8").replace(
                "\r\n", "\n"
            ) != expected_python.replace("\r\n", "\n"):
                errors.append(f"选修双格式实验漂移: {python_path.relative_to(ROOT)}")
        else:
            python_path.write_text(expected_python, encoding="utf-8")

    for error in errors:
        print(f"错误: {error}")
    if errors:
        return 1
    print("实验契约与 ipynb/py 双格式同步通过。" if check else "实验双格式已同步。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查，不写入文件")
    args = parser.parse_args()
    return sync_labs(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
