"""Greedy parity harness: flatrun vs an OpenAI-compatible /v1/completions
endpoint (LM Studio, llama-server, etc.).

The CLI's ``--no-sample`` already uses argmax, so this script just
runs it once, grabs the printed text, and asks the server for the
greedy continuation of the same prompt. The two should match exactly
on the first token and usually for many more; a divergence on the
first token almost always means a weight/RoPE/orientation bug.

Usage::

    python tools/compare_to_lmstudio.py model.gguf --prompt "Once upon a time"
    python tools/compare_to_lmstudio.py model.gguf --endpoint http://localhost:1234
    python tools/compare_to_lmstudio.py model.gguf --server-model qwen2.5-coder-0.5b -n 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

LMSTUDIO_DEFAULT = "http://192.168.1.7:1234"


def _run_flatrun(path: Path, prompt: str, n: int) -> str:
    r = subprocess.run(
        [
            sys.executable, "-m", "flatrun.cli",
            "--model", str(path),
            "--prompt", prompt,
            "--no-chat-template",
            "--max-new", str(n),
            "--no-sample", "--top-k", "0",
        ],
        check=True, capture_output=True, text=True,
        env={"PYTHONPATH": "src", **os.environ},
    )
    return r.stdout.split("Generated text: '", 1)[-1].split("'\n", 1)[0]


def _run_server(endpoint: str, model: str, prompt: str, n: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0,
            "top_k": 1,
            "repeat_penalty": 1.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["text"]


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--endpoint", default=LMSTUDIO_DEFAULT)
    p.add_argument("--server-model", default=None)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("-n", type=int, default=12)
    p.add_argument("--allow-diff", type=int, default=0)
    args = p.parse_args()

    server = args.server_model or args.model.stem
    ours = _run_flatrun(args.model, args.prompt, args.n)
    theirs = _run_server(args.endpoint, server, args.prompt, args.n)
    n_match = _common_prefix_len(ours, theirs)
    # "Agree" means: the shorter output is a prefix of the longer one
    # (the longer side just kept going past the shorter side's max_new).
    shorter, longer = sorted([ours, theirs], key=len)
    agreed = longer.startswith(shorter)
    summary = {
        "model": args.model.name,
        "prompt": args.prompt,
        "n": args.n,
        "ours": ours,
        "theirs": theirs,
        "common_prefix_chars": n_match,
        "agreed": agreed,
    }
    print(json.dumps(summary, indent=2))
    return 0 if agreed else 1


if __name__ == "__main__":
    raise SystemExit(main())
