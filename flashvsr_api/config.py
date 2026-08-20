from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_list(name: str, default: str) -> tuple[Path, ...]:
    raw = os.getenv(name, default)
    items = [item.strip() for item in raw.split(":") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one path")
    return tuple(Path(item).expanduser().resolve() for item in items)


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    runtime_root: Path = Path(
        os.getenv("FLASHVSR_RUNTIME_ROOT", REPO_ROOT / "runtime/api")
    ).expanduser().resolve()
    model_root: Path = Path(
        os.getenv(
            "FLASHVSR_MODEL_ROOT",
            REPO_ROOT / "examples/WanVSR/FlashVSR-v1.1",
        )
    ).expanduser().resolve()
    v1_model_root: Path = Path(
        os.getenv(
            "FLASHVSR_V1_MODEL_ROOT",
            REPO_ROOT / "examples/WanVSR/FlashVSR",
        )
    ).expanduser().resolve()
    allowed_input_roots: tuple[Path, ...] = _path_list(
        "FLASHVSR_ALLOWED_INPUT_ROOTS", os.fspath(REPO_ROOT)
    )
    device: str = os.getenv("FLASHVSR_DEVICE", "0")
    max_frames: int = int(os.getenv("FLASHVSR_MAX_FRAMES", "300"))
    max_target_pixels: int = int(
        os.getenv("FLASHVSR_MAX_TARGET_PIXELS", "2000000")
    )

    @property
    def jobs_dir(self) -> Path:
        return self.runtime_root / "jobs"

    @property
    def outputs_dir(self) -> Path:
        return self.runtime_root / "outputs"

    def model_root_for(self, version: str) -> Path:
        if version == "v1":
            return self.v1_model_root
        if version == "v1.1":
            return self.model_root
        raise ValueError(f"Unsupported model version: {version}")
