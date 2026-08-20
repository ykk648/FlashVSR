# FlashVSR FastAPI service

The service accepts local filesystem paths, keeps one Tiny-Long pipeline
resident, and serializes inference through one GPU worker. It supports both
FlashVSR v1 and v1.1.

## Start

Install the environment and models using `UV_DEPLOYMENT.md`, then run:

```bash
set -a
. ./.env
set +a
uv run flashvsr-api --host 127.0.0.1 --port 18302 --device 0
```

Do not use multiple uvicorn workers. Each worker would load another model onto
the same GPU. API documentation is available at
`http://127.0.0.1:18302/docs`.

## Submit a job

`input_path` must be inside one of the colon-separated directories in
`FLASHVSR_ALLOWED_INPUT_ROOTS`.

```bash
curl -sS http://127.0.0.1:18302/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"input_path":"/data/videos/input.mp4","scale":1.4,"model_version":"v1.1"}'
```

Inspect, cancel, or download a job:

```bash
curl -sS http://127.0.0.1:18302/api/jobs/JOB_ID
curl -sS -X POST http://127.0.0.1:18302/api/jobs/JOB_ID/cancel
curl -OJ http://127.0.0.1:18302/api/jobs/JOB_ID/download
```

Queued cancellation is immediate. Active cancellation is checked between
preprocessing, inference, and encoding because the upstream CUDA pipeline does
not expose per-block interruption.

## API endpoints

- `POST /api/jobs`: create a path-based super-resolution job.
- `GET /api/jobs/{job_id}`: read persisted status and metadata.
- `POST /api/jobs/{job_id}/cancel`: request cancellation.
- `GET /api/jobs/{job_id}/download`: download a completed MP4.
- `GET /healthz`: process liveness.
- `GET /readyz`: worker, GPU selection, and resident model state.

Jobs move through `queued`, `running`, and one of `completed`, `failed`, or
`cancelled`. Metadata is written atomically under `runtime/api/jobs`.

## Limits and output

Defaults are 300 source frames and 2 million target pixels. They are empirical
RTX 4090 limits for Tiny-Long and can be overridden with
`FLASHVSR_MAX_FRAMES` and `FLASHVSR_MAX_TARGET_PIXELS`. A 1376x768 source at
scale 1.4 becomes 1920x1024 after the model's 128-pixel alignment.

The service preserves the exact rational frame rate and source frame count,
uses NVENC with libx264 fallback, and remuxes the original audio stream.
Switching `model_version` between `v1` and `v1.1` unloads the current resident
pipeline before loading the requested weights.

## Production boundary

The API has no authentication and accepts server-local paths. Keep it bound to
localhost or place it behind an authenticated reverse proxy. Do not expose it
directly to an untrusted network. Use separate allowlisted input and runtime
directories when processing files from other users.
