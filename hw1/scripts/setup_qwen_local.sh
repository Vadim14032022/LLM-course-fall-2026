#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mapfile -t RUNTIME_VALUES < <(python - "$ROOT_DIR/runtime.json" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
print(runtime["software"]["conda_environment"])
print(runtime["model"]["id"])
print(runtime["model"]["revision"])
print(runtime["software"]["vllm"])
PY
)
ENV_NAME="${RUNTIME_VALUES[0]}"
MODEL_ID="${RUNTIME_VALUES[1]}"
MODEL_REVISION="${RUNTIME_VALUES[2]}"
CUDA_BACKEND="${RUNTIME_VALUES[3]##*+}"

if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  conda env update -n "$ENV_NAME" --file "$ROOT_DIR/configs/environment.yml"
else
  conda env create -n "$ENV_NAME" --file "$ROOT_DIR/configs/environment.yml"
fi

conda run -n "$ENV_NAME" python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
if [[ ! -e "$ENV_PREFIX/lib64" ]]; then
  ln -s targets/x86_64-linux/lib "$ENV_PREFIX/lib64"
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install 'uv==0.13.7'
conda run --no-capture-output -n "$ENV_NAME" uv pip install \
  --python "$ENV_PREFIX/bin/python" --torch-backend="$CUDA_BACKEND" \
  --index-strategy=unsafe-best-match --requirement "$ROOT_DIR/requirements.txt"
conda run -n "$ENV_NAME" python -m pip check

# Download the pinned model and tokenizer before offline inference.
conda run --no-capture-output -n "$ENV_NAME" python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', revision='$MODEL_REVISION')"

echo "Client, PDF parser and Qwen server are ready in $ENV_NAME (Python 3.11)."
echo "Activate it with: conda activate $ENV_NAME"
echo "Start the API with: python '$ROOT_DIR/scripts/serve_qwen_local.py'"
