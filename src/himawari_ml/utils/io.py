from __future__ import annotations

import os
from pathlib import Path

def project_root() -> Path:
    # assumes this file is src/himawari_ml/utils/io.py
    return Path(__file__).resolve().parents[3]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def data_dir() -> Path:
    return project_root() / "data"

def outputs_dir() -> Path:
    return project_root() / "outputs"
