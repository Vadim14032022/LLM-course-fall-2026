#!/usr/bin/env python3
"""Start the local Qwen OpenAI-compatible API server."""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "runtime.json"


def build_command(config: dict) -> list[str]:
    model = config["model"]
    api = config["api"]
    resources = config["resources"]
    endpoint = urlsplit(api["base_url"])
    command = [
        "vllm",
        "serve",
        model["id"],
        "--revision",
        model["revision"],
        "--host",
        endpoint.hostname or "127.0.0.1",
        "--port",
        str(endpoint.port or (443 if endpoint.scheme == "https" else 80)),
        "--dtype",
        model["precision"],
        "--max-model-len",
        str(resources["max_model_len"]),
        "--max-num-seqs",
        str(resources["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(resources["gpu_memory_utilization"]),
        "--reasoning-parser",
        resources["reasoning_parser"],
    ]
    if resources.get("language_model_only", False):
        command.append("--language-model-only")
    api_key = os.environ.get(api["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Set {api['api_key_env']} before launching the local API server."
        )
    command.extend(["--api-key", api_key])
    return command


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "--print-command" in sys.argv[1:]:
        try:
            command = build_command(config)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 2
        # Do not display the API key in diagnostic output.
        if "--api-key" in command:
            index = command.index("--api-key") + 1
            command[index] = f"<{config['api']['api_key_env']}>"
        print(" ".join(command))
        return 0
    try:
        result = subprocess.run(build_command(config), check=False)
    except FileNotFoundError:
        print(
            "vLLM is not installed in this environment. Run scripts/setup_qwen_local.sh first.",
            file=sys.stderr,
        )
        return 127
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
