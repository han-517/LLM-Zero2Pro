from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_starter(filename: str) -> ModuleType:
    path = ROOT / "exercises" / "starter" / filename
    spec = importlib.util.spec_from_file_location(f"student_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载练习模板: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
