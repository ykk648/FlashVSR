#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import shutil
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the FlashVSR API runtime.")
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail instead of warn when CUDA is unavailable."
    )
    args = parser.parse_args()
    failures: list[str] = []

    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 required, found {sys.version.split()[0]}")

    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            failures.append(f"Missing executable: {command}")

    for module_name in ("fastapi", "uvicorn", "block_sparse_attn"):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failures.append(f"Cannot import {module_name}: {error}")

    try:
        import torch

        print(f"torch={torch.__version__}")
        if not torch.__version__.startswith("2.6.0+"):
            failures.append(f"Expected Torch 2.6.0 CUDA wheel, found {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"cuda_available={cuda_available}")
        if args.require_cuda and not cuda_available:
            failures.append("CUDA is unavailable")
    except Exception as error:
        failures.append(f"Cannot import torch: {error}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("FlashVSR deployment check passed")


if __name__ == "__main__":
    main()
