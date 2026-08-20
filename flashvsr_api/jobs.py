from __future__ import annotations

import json
import logging
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .engine import FlashVSREngine
from .video import probe_video, target_dimensions


LOGGER = logging.getLogger(__name__)
FINAL_STATES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = FlashVSREngine(settings)
        self.jobs: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        queued: list[str] = []
        for path in self.settings.jobs_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("Ignoring unreadable job file %s", path)
                continue
            if record.get("status") == "running":
                record.update(
                    status="failed",
                    stage="failed",
                    error="API process restarted while the job was running",
                    updated_at=utc_now(),
                    finished_at=utc_now(),
                )
                self._persist(record)
            elif record.get("status") == "queued":
                queued.append(record["id"])
            self.jobs[record["id"]] = record

        self._thread = threading.Thread(
            target=self._worker, name="flashvsr-gpu-worker", daemon=True
        )
        self._thread.start()
        for job_id in queued:
            self._queue.put(job_id)

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _validate_path(self, raw_path: str, *, must_exist: bool) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in self.settings.allowed_input_roots):
            roots = ", ".join(os.fspath(root) for root in self.settings.allowed_input_roots)
            raise ValueError(f"Path is outside allowed roots: {roots}")
        if must_exist and not path.is_file():
            raise ValueError(f"Input video does not exist: {path}")
        return path

    def create(
        self,
        input_path: str,
        scale: float,
        output_path: str | None,
        model_version: str = "v1.1",
    ) -> dict[str, Any]:
        self.settings.model_root_for(model_version)
        source = self._validate_path(input_path, must_exist=True)
        info = probe_video(source)
        if info.frames > self.settings.max_frames:
            raise ValueError(
                f"Input has {info.frames} frames; limit is {self.settings.max_frames}"
            )
        width, height = target_dimensions(info.width, info.height, scale)
        pixels = width * height
        if pixels > self.settings.max_target_pixels:
            raise ValueError(
                f"Target {width}x{height} has {pixels} pixels; "
                f"limit is {self.settings.max_target_pixels}"
            )

        job_id = uuid.uuid4().hex
        if output_path:
            output = self._validate_path(output_path, must_exist=False)
            if output.suffix.lower() != ".mp4":
                raise ValueError("output_path must end in .mp4")
            if output.exists():
                raise ValueError(f"Output already exists: {output}")
        else:
            output = self.settings.outputs_dir / f"{job_id}.mp4"
        if output == source:
            raise ValueError("Input and output paths must differ")

        now = utc_now()
        record: dict[str, Any] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "input_path": os.fspath(source),
            "output_path": os.fspath(output),
            "scale": scale,
            "model_version": model_version,
            "source": info.to_dict(),
            "target": {"width": width, "height": height},
            "encoder": None,
            "cancel_requested": False,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            self.jobs[job_id] = record
            self._persist(record)
        self._queue.put(job_id)
        return dict(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.jobs.get(job_id)
            return dict(record) if record else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.jobs.get(job_id)
            if record is None:
                return None
            if record["status"] in FINAL_STATES:
                return dict(record)
            record["cancel_requested"] = True
            record["updated_at"] = utc_now()
            if record["status"] == "queued":
                record.update(
                    status="cancelled", stage="cancelled", finished_at=utc_now()
                )
            self._persist(record)
            return dict(record)

    def _persist(self, record: dict[str, Any]) -> None:
        path = self.settings.jobs_dir / f"{record['id']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            record = self.jobs[job_id]
            record.update(values, updated_at=utc_now())
            self._persist(record)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return bool(self.jobs[job_id]["cancel_requested"])

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                record = self.jobs.get(job_id)
                if record is None or record["status"] != "queued":
                    continue
                record.update(
                    status="running", stage="loading_model", started_at=utc_now(), updated_at=utc_now()
                )
                self._persist(record)

            try:
                source = Path(record["input_path"])
                output = Path(record["output_path"])
                info = probe_video(source)
                encoder = self.engine.upscale(
                    input_path=source,
                    output_path=output,
                    info=info,
                    model_version=record.get("model_version", "v1.1"),
                    scale=record["scale"],
                    target_width=record["target"]["width"],
                    target_height=record["target"]["height"],
                    progress=lambda stage: self._update(job_id, stage=stage),
                    cancelled=lambda: self._is_cancelled(job_id),
                )
                self._update(
                    job_id,
                    status="completed",
                    stage="completed",
                    encoder=encoder,
                    output_bytes=output.stat().st_size,
                    finished_at=utc_now(),
                )
            except InterruptedError as error:
                Path(record["output_path"]).unlink(missing_ok=True)
                self._update(
                    job_id,
                    status="cancelled",
                    stage="cancelled",
                    error=str(error),
                    finished_at=utc_now(),
                )
            except BaseException as error:
                LOGGER.error("FlashVSR job %s failed\n%s", job_id, traceback.format_exc())
                Path(record["output_path"]).unlink(missing_ok=True)
                self._update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=f"{type(error).__name__}: {error}",
                    finished_at=utc_now(),
                )
