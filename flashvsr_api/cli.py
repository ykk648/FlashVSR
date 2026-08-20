from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FlashVSR FastAPI service.")
    parser.add_argument(
        "--host", default=os.getenv("FLASHVSR_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("FLASHVSR_PORT", "18302"))
    )
    parser.add_argument(
        "--device", default=os.getenv("FLASHVSR_DEVICE", "0"), help="Physical GPU index."
    )
    args = parser.parse_args()

    os.environ["FLASHVSR_DEVICE"] = args.device
    import uvicorn

    uvicorn.run(
        "flashvsr_api.main:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
