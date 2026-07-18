from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_course.paths import PROJECT_ROOT
from llm_course.schemas import ValidationReport


@dataclass(frozen=True)
class Exercise:
    exercise_id: str
    slug: str
    week: str
    title: str
    template: str
    check: str


EXERCISES = (
    Exercise(
        "01",
        "softmax",
        "4",
        "数值稳定 Softmax",
        "exercises/starter/01_stable_softmax.py",
        "exercises/checks/test_01_stable_softmax.py",
    ),
    Exercise(
        "02",
        "attention",
        "12–13",
        "单头因果注意力",
        "exercises/starter/02_causal_attention.py",
        "exercises/checks/test_02_causal_attention.py",
    ),
    Exercise(
        "03",
        "kv-cache",
        "20",
        "KV Cache 计算与存储",
        "exercises/starter/03_kv_cache_budget.py",
        "exercises/checks/test_03_kv_cache_budget.py",
    ),
    Exercise(
        "04",
        "sft",
        "40",
        "SFT next-token 对齐",
        "exercises/starter/04_sft_shift.py",
        "exercises/checks/test_04_sft_shift.py",
    ),
    Exercise(
        "05",
        "moe-capacity",
        "35–36",
        "MoE 容量与溢出",
        "exercises/starter/05_moe_capacity.py",
        "exercises/checks/test_05_moe_capacity.py",
    ),
    Exercise(
        "06",
        "bpe",
        "9–10",
        "Byte BPE 合并",
        "exercises/starter/06_byte_bpe.py",
        "exercises/checks/test_06_byte_bpe.py",
    ),
    Exercise(
        "07",
        "rope",
        "21–22",
        "RoPE 旋转位置编码",
        "exercises/starter/07_rope.py",
        "exercises/checks/test_07_rope.py",
    ),
    Exercise(
        "08",
        "gqa",
        "23–24",
        "Grouped-Query Attention",
        "exercises/starter/08_grouped_query_attention.py",
        "exercises/checks/test_08_grouped_query_attention.py",
    ),
    Exercise(
        "09",
        "decoder",
        "21–22",
        "RMSNorm 与 SwiGLU",
        "exercises/starter/09_modern_decoder.py",
        "exercises/checks/test_09_modern_decoder.py",
    ),
    Exercise(
        "10",
        "moe-router",
        "35–36",
        "Top-k MoE 路由",
        "exercises/starter/10_moe_router.py",
        "exercises/checks/test_10_moe_router.py",
    ),
)


def select_exercises(identifier: str) -> tuple[Exercise, ...]:
    normalized = identifier.strip().lower().replace("_", "-")
    if normalized == "all":
        return EXERCISES
    for exercise in EXERCISES:
        aliases = {
            exercise.exercise_id,
            exercise.slug,
            Path(exercise.template).stem.lower().replace("_", "-"),
        }
        if normalized in aliases:
            return (exercise,)
    choices = ", ".join(exercise.exercise_id for exercise in EXERCISES)
    raise ValueError(f"未知练习 {identifier!r}；可用编号为 {choices}，或使用 all")


def _template_state(exercise: Exercise) -> str:
    path = PROJECT_ROOT / exercise.template
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return "缺失"
    return "待填写" if "raise NotImplementedError" in source else "已填写，待核查"


def print_exercises() -> int:
    print("编号  周次   状态             核心代码模板")
    for exercise in EXERCISES:
        print(
            f"{exercise.exercise_id:<4}  {exercise.week:<5}  "
            f"{_template_state(exercise):<15}  {exercise.title} ({exercise.slug})"
        )
    print("\n核查示例: uv run llm-course exercises check 07")
    return 0


def run_exercise_checks(identifier: str = "all") -> int:
    try:
        selected = select_exercises(identifier)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2
    paths = [str(PROJECT_ROOT / exercise.check) for exercise in selected]
    names = ", ".join(exercise.exercise_id for exercise in selected)
    print(f"开始核查练习: {names}")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode == 0:
        print("练习核查通过。请再口述张量形状、边界条件和复杂度。")
    else:
        print("练习尚未通过；从第一条失败信息和最小反例开始修正。")
    return completed.returncode


def validate_exercise_assets() -> ValidationReport:
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for exercise in EXERCISES:
        if exercise.exercise_id in seen_ids:
            report.errors.append(f"练习编号重复: {exercise.exercise_id}")
        if exercise.slug in seen_slugs:
            report.errors.append(f"练习别名重复: {exercise.slug}")
        seen_ids.add(exercise.exercise_id)
        seen_slugs.add(exercise.slug)
        for relative_path in (exercise.template, exercise.check):
            path = PROJECT_ROOT / relative_path
            if not path.is_file():
                report.errors.append(f"缺少练习资产: {relative_path}")
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, SyntaxError) as exc:
                report.errors.append(f"练习资产无法编译: {relative_path}: {exc}")
    return report
