#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

from modelscope import snapshot_download


MODEL_ID = "kuohao/FlashVSR-v1.1"
MODEL_FILES = {
    "diffusion_pytorch_model_streaming_dmd.safetensors": (
        5_676_070_392,
        "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be",
    ),
    "LQ_proj_in.ckpt": (
        575_694_948,
        "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2",
    ),
    "TCDecoder.ckpt": (
        189_018_333,
        "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
    ),
    "Wan2.1_VAE.pth": (
        507_609_880,
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(local_dir: Path) -> None:
    for name, (expected_size, expected_sha256) in MODEL_FILES.items():
        path = local_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {name}: {actual_size} != {expected_size}")
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual_sha256}")
        print(f"verified {name} ({actual_size} bytes)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download and verify FlashVSR v1.1 from ModelScope.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=repo_root / "examples" / "WanVSR" / "FlashVSR-v1.1",
    )
    parser.add_argument("--cache-dir", type=Path, default=repo_root / ".modelscope-cache")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        snapshot_download(
            MODEL_ID,
            revision="master",
            cache_dir=str(args.cache_dir),
            local_dir=str(args.local_dir),
            allow_file_pattern=list(MODEL_FILES),
            max_workers=4,
        )
    verify(args.local_dir)


if __name__ == "__main__":
    main()
