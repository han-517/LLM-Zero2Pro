from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (
    ROOT / "learning" / "labs" / "projects" / "01_end_to_end_lm",
    ROOT / "learning" / "labs" / "projects" / "02_real_data_pipeline",
    ROOT / "learning" / "labs" / "projects" / "03_gpu_systems",
    ROOT / "learning" / "labs" / "projects" / "04_scaling_laws",
    ROOT / "learning" / "labs" / "projects" / "05_alignment_rl",
)

for project in reversed(PROJECTS):
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
