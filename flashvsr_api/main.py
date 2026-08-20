from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .jobs import JobManager


settings = Settings()
manager = JobManager(settings)


class CreateJobRequest(BaseModel):
    input_path: str = Field(min_length=1)
    scale: float = Field(default=1.4, ge=1, le=4)
    output_path: str | None = None
    model_version: Literal["v1", "v1.1"] = "v1.1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(
    title="FlashVSR API",
    version="1.0.0",
    description="Single-GPU queued video super-resolution service.",
    lifespan=lifespan,
)


def require_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready() -> dict[str, object]:
    if not manager.worker_alive:
        raise HTTPException(status_code=503, detail="GPU worker is not running")
    return {
        "status": "ready",
        "model_loaded": manager.engine.loaded,
        "loaded_model_version": manager.engine.loaded_version,
        "physical_gpu": settings.device,
    }


@app.post("/api/jobs", status_code=202)
def create_job(request: CreateJobRequest) -> dict:
    try:
        return manager.create(
            request.input_path,
            request.scale,
            request.output_path,
            request.model_version,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return require_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    job = require_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")
    output = Path(job["output_path"])
    if not output.is_file():
        raise HTTPException(status_code=410, detail="Output file is missing")
    return FileResponse(output, media_type="video/mp4", filename=output.name)
