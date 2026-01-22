from __future__ import annotations

from pathlib import Path
from PIL import Image

def validate_image(path: Path) -> None:
    img = Image.open(path)
    img.verify()  # raises if corrupted
