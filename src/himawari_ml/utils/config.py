from __future__ import annotations
import certifi
CA_BUNDLE = certifi.where()

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    sample_url: str | None = os.getenv("HIMAWARI_SAMPLE_URL") or None
    image_size: int = int(os.getenv("IMAGE_SIZE", "512"))
    seed: int = int(os.getenv("SEED", "42"))

SETTINGS = Settings()
