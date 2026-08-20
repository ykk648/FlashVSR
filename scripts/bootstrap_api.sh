#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-.uv-python}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

uv sync --extra build --extra api

wheel_name="block_sparse_attn-0.0.2+cu124torch2.6sm80ptx1-cp311-cp311-linux_x86_64.whl"
wheel_path="${BLOCK_SPARSE_ATTN_WHEEL:-dist/${wheel_name}}"
expected_sha256="1a8f3e7bf45f25dba2c2df83c9ecf13601b27ea5cdcea6e4283a1539e72f8c7d"

if [[ ! -f "${wheel_path}" ]]; then
    printf 'Missing Block-Sparse-Attention wheel: %s\n' "${wheel_path}" >&2
    printf 'Download the release asset, then set BLOCK_SPARSE_ATTN_WHEEL to its path.\n' >&2
    exit 1
fi

printf '%s  %s\n' "${expected_sha256}" "${wheel_path}" | sha256sum --check --status
uv pip install --python .venv/bin/python --reinstall --no-deps "${wheel_path}"

.venv/bin/python scripts/check_deployment.py
printf 'Environment ready. Download models next; see UV_DEPLOYMENT.md.\n'
