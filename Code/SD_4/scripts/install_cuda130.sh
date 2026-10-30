#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: activate a dedicated Conda/virtual environment first." >&2
  exit 2
fi

"${PYTHON_BIN}" -c 'import platform, sys; assert sys.version_info[:2] == (3, 10), f"expected Python 3.10.x, found {sys.version}"; assert platform.system() == "Linux" and platform.machine() == "x86_64", f"expected Linux x86_64, found {platform.platform()}"'

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install "uv==0.13.1"

UV_BIN="$(dirname -- "$(command -v "${PYTHON_BIN}")")/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv)"
fi

"${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  --torch-backend=cu130 \
  -r "${PROJECT_DIR}/requirements-vllm.txt"

"${PYTHON_BIN}" -m pip check
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/check_cuda130_env.py"
