# FlashVSR uv deployment

This fork pins a tested Linux GPU runtime:

- CPython 3.11
- PyTorch 2.6.0 with CUDA 12.4 wheels
- Block-Sparse-Attention 0.0.2 built for CPython 3.11 and the Torch 2.6 ABI
- NVIDIA driver with CUDA 12.4 runtime support or newer
- `ffmpeg` and `ffprobe` on `PATH`

Run commands from the repository root.

## 1. Install uv and the wheel asset

Install [uv](https://docs.astral.sh/uv/) and download this repository's
Block-Sparse-Attention GitHub Release asset into `dist/`:

```text
dist/block_sparse_attn-0.0.2+cu124torch2.6sm80ptx1-cp311-cp311-linux_x86_64.whl
```

The wheel SHA-256 must be:

```text
1a8f3e7bf45f25dba2c2df83c9ecf13601b27ea5cdcea6e4283a1539e72f8c7d
```

It requires Linux x86_64, glibc 2.32 or newer, and
`GLIBCXX_3.4.29`. The target machine does not need a CUDA toolkit or `nvcc`.

## 2. Bootstrap the API environment

```bash
scripts/bootstrap_api.sh
```

The script runs `uv sync --extra build --extra api`, verifies the wheel hash,
installs the extension with `--no-deps`, and checks the runtime imports. To use
a wheel stored elsewhere:

```bash
BLOCK_SPARSE_ATTN_WHEEL=/path/to/block_sparse_attn.whl \
  scripts/bootstrap_api.sh
```

`uv sync` intentionally does not manage the platform-specific CUDA extension.
Run the bootstrap script again after any future sync.

## 3. Download models from ModelScope

Download v1.1, the default API model:

```bash
.venv/bin/python scripts/download_flashvsr_v1_1.py
```

Optionally download v1 for A/B comparisons:

```bash
.venv/bin/python scripts/download_flashvsr_v1.py
```

The scripts download only required inference files and verify exact sizes and
upstream LFS SHA-256 values. Rerun verification without network access using:

```bash
.venv/bin/python scripts/download_flashvsr_v1_1.py --verify-only
.venv/bin/python scripts/download_flashvsr_v1.py --verify-only
```

For networks that require a proxy, set `http_proxy`, `https_proxy`, and
`ALL_PROXY` only for the download command.

## 4. Configure and start

```bash
cp .env.example .env
# Edit FLASHVSR_ALLOWED_INPUT_ROOTS and the GPU index.
set -a
. ./.env
set +a
uv run flashvsr-api
```

The default bind address is `127.0.0.1:18302`. Keep exactly one API process per
GPU. See `API_SERVICE.md` for endpoints and production notes.

## Build Block-Sparse-Attention from source

The prebuilt wheel came from upstream commit
`49d6c39e4dc0303442cda3bb758b3925d4399c49`. To reproduce it:

```bash
git clone --recurse-submodules \
  https://github.com/mit-han-lab/Block-Sparse-Attention.git \
  ../Block-Sparse-Attention
git -C ../Block-Sparse-Attention checkout 49d6c39e4dc0303442cda3bb758b3925d4399c49
git -C ../Block-Sparse-Attention submodule update --init --recursive
git -C ../Block-Sparse-Attention apply \
  "$PWD/patches/block-sparse-attention-import-order.patch"

CUDA_HOME=/usr/local/cuda-12.8 \
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE \
BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=8 NVCC_THREADS=4 \
UV_CACHE_DIR=.uv-cache \
uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
  ../Block-Sparse-Attention
```

The upstream BSD license is preserved in
`third_party/Block-Sparse-Attention-LICENSE`.
