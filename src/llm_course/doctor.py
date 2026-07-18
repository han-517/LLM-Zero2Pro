from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from llm_course.paths import PROJECT_ROOT


def collect_diagnostics() -> dict[str, object]:
    seed = 20260718
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0], [1.0]])
    product = left @ right
    expected = torch.tensor([[4.0], [10.0]])

    data_dir = PROJECT_ROOT / "data"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "project_root_exists": PROJECT_ROOT.is_dir(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": seed,
        "matmul_ok": bool(torch.allclose(product, expected)),
        "matmul_result": product.flatten().tolist(),
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.is_dir(),
    }


def run_doctor(as_json: bool = False) -> int:
    info = collect_diagnostics()
    if as_json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print("LLM 课程环境检查")
        print("=" * 30)
        for key, value in info.items():
            print(f"{key:20} {value}")
        if not info["data_dir_exists"]:
            print("提示：data/ 会在第一次运行数据实验时创建；这不是错误。")
    return 0 if info["matmul_ok"] and Path(info["project_root"]).is_dir() else 1

