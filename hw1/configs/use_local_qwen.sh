#!/usr/bin/env bash
# Source this file in the client shell while developing against local Qwen.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mapfile -t RUNTIME_VALUES < <(python - "$ROOT_DIR/runtime.json" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
print(runtime["api"]["base_url"])
print(runtime["model"]["id"])
print(runtime["api"]["local_development_submission_id"])
PY
)
export LLM_BASE_URL="${LLM_BASE_URL:-${RUNTIME_VALUES[0]}}"
export LLM_MODEL="${LLM_MODEL:-${RUNTIME_VALUES[1]}}"
export SUBMISSION_ID="${SUBMISSION_ID:-${RUNTIME_VALUES[2]}}"
