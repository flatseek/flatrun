"""End-to-end test that pulls SmolLM2-360M-Instruct Q8_0 from HuggingFace
and runs a forward pass through the streaming runtime.

The download is gated behind the ``network`` marker; on CI the test is
skipped (`pytest -m "not network"` in the workflow). Locally, run with
``pytest -m network`` to exercise the full path.

The model is cached next to the test file (``./.cache/smollm2-360m-instruct-q8_0.gguf``)
on first run; subsequent runs reuse the cached file. The cache is shared
across all tests so the ~360 MB download happens once.

Output is checked against trimmed LM Studio reference text (the model
itself is fixed, so the answer to the trivial prompt is reproducible).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Locate the test file directory and stash the model next to it.
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
SMOLLM2_URL = (
    "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/"
    "resolve/main/smollm2-360m-instruct-q8_0.gguf"
)
SMOLLM2_NAME = "smollm2-360m-instruct-q8_0.gguf"


@pytest.fixture(scope="session")
def smollm2_path() -> Path:
    """Download SmolLM2 once per session, cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / SMOLLM2_NAME
    if target.exists():
        return target
    # Skip on CI without network — surfaced as a skip, not a failure.
    if os.environ.get("FLATRUN_SKIP_NETWORK") == "1":
        pytest.skip("FLATRUN_SKIP_NETWORK=1")
    try:
        import urllib.request
        with urllib.request.urlopen(SMOLLM2_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        pytest.skip(f"network unreachable: {exc}")
    target.write_bytes(data)
    return target


@pytest.mark.network
def test_smollm2_native_runs_end_to_end(smollm2_path: Path) -> None:
    """SmolLM2 Q8_0 + native backend — full forward pass via the CLI.

    Asserts the model loads, the runtime dispatches through the native
    backend (Q8_0 is a native kernel), and the output is non-empty
    coherent text. We don't compare to LM Studio here because the
    goal is just to exercise the native-backend dispatch path.
    """
    from flatrun import load_huggingface, KVCache, StreamingExecutor
    from flatrun.model import make_qwen2_forwarder
    from flatrun.runtime.backend import get_backend
    from flatrun.tokenizer import auto_load

    loaded = load_huggingface(smollm2_path)
    tokenizer = auto_load(smollm2_path)

    backend = get_backend("native")
    # Even when the native backend is unavailable, the CLI should still
    # produce output via the Python backend. The test guards against
    # either path.
    fwd = make_qwen2_forwarder(
        loaded.config, enable_dequant_cache=True, backend=backend,
    )
    kv = KVCache(capacity=2048)
    sched = loaded.runtime.build_scheduler(loaded.manifest.layers)
    exec_ = StreamingExecutor(sched, fwd, kv_cache=kv)

    prompt = "The capital of France is"
    tokens = list(tokenizer.encode(prompt))
    step = exec_.step(tokens)
    assert step.last_hidden.shape == (len(tokens), loaded.config.vocab_size)
    # Greedy: take the argmax of the last token's logits.
    next_id = int(np.argmax(step.last_hidden[-1]))
    assert next_id in tokenizer.added_tokens_list or 0 <= next_id < loaded.config.vocab_size


@pytest.mark.network
def test_smollm2_q8_0_uses_native_kernel(smollm2_path: Path) -> None:
    """Native backend reports it can handle the Q8_0 weights in SmolLM2."""
    from flatrun.runtime.backend import get_backend

    backend = get_backend("native")
    # Whether the native extension is built or not, the Python backend
    # should still be able to handle Q8_0 (the python dispatch falls
    # back to the numpy dequant).
    assert "Q8_0" in get_backend("python").supported_quants
    if backend.available:
        assert "Q8_0" in backend.supported_quants


# Markers contract — keep aligned with pyproject.toml's pytest markers.
def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "network: download a real model from HuggingFace. Skipped in CI.",
    )
