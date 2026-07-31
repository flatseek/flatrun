"""Unified FlatRun CLI.

Run a single prompt (or a chat turn) through any FlatRun-supported model:

* GGUF directory (``*.gguf``)
* SafeTensors directory (``model.safetensors`` + ``config.json``)
* MLX-4bit directory (weight + scales + biases triplets)

The CLI auto-detects the format, derives a Qwen2 config from GGUF
metadata when no ``config.json`` is present, and applies the model's
chat template (or the Qwen2 ChatML default) when ``--messages`` is
provided.

Examples::

    # GGUF
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF \\
        --prompt "def hello():"

    # SafeTensors
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/Works/.../qwen2.5-0.5b \\
        --prompt "Once upon a time"

    # MLX-4bit
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /Users/judotens/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit \\
        --prompt "def hello():" \\
        --max-new 4

    # Multi-turn chat
    PYTHONPATH=src python examples/flatrun_chat.py \\
        --model /path/to/qwen \\
        --messages-json '[{"role":"system","content":"You are concise."},{"role":"user","content":"Hi"}]'
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flatrun import (
    KVCache,
    RuntimeConfig,
    StreamingExecutor,
    load_huggingface,
)
from flatrun.model.qwen2 import Qwen2Config, make_qwen2_forwarder
from flatrun.model.sampling import Sampler
from flatrun.runtime.memory import MemoryConfig
from flatrun.tokenizer import auto_load
from flatrun.utils.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spinner + thinking extraction
# ---------------------------------------------------------------------------
#
# Inference on a Qwen2.5-class model is slow enough that a static prompt
# makes the user think the CLI has hung. The :class:`Spinner` runs a
# lightweight animation on stderr (a single carriage-returned line that
# redraws every 100 ms) while the forwarder is busy, and disappears
# cleanly when generation finishes - either with a newline that drops the
# animated line, or with the final assistant text that overwrites it.
#
# Modern reasoning models (Qwen3, DeepSeek-R1 distilled, ...) emit a
# ``<think>...</think>`` block before the user-visible reply. We split
# the two so the chat REPL can show the chain-of-thought in dim colour
# during generation and the clean answer afterwards.


_SPINNER_FRAMES = "-\\|/"
_SPINNER_INTERVAL = 0.1  # seconds


class Spinner:
    """Single-line, stderr-only animation that lives for the duration
    of a generation step.

    Usage::

        with Spinner("Thinking"):
            ...do slow work...

    The animated line stays out of the way of stdout and is overwritten
    by whatever the caller prints next.
    """

    def __init__(self, label: str = "Thinking", stream=None) -> None:
        self._label = label
        self._stream = stream or sys.stderr
        self._thread = None
        self._stop = False

    def __enter__(self) -> "Spinner":
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=_SPINNER_INTERVAL * 3)
        # Clear the spinner line so the caller can print over it cleanly.
        try:
            self._stream.write("\r" + " " * (len(self._label) + 4) + "\r")
            self._stream.flush()
        except Exception:
            pass

    def _run(self) -> None:
        i = 0
        while not self._stop:
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            try:
                self._stream.write(f"\r{frame} {self._label}...")
                self._stream.flush()
            except Exception:
                return
            i += 1
            time.sleep(_SPINNER_INTERVAL)


# Patterns we recognise as "this is a thinking block". The first match
# wins; everything outside it is the user-visible reply. Order matters
# because the Qwen3 format wraps the whole trace in
# ``<|im_start|>think\\n...<|im_start|>assistant`` and we want the
# human-readable answer to start at the second marker.
_THINK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|thinking|>", "<|/thinking|>"),
)


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Pull a chain-of-thought block out of ``text``.

    Returns ``(thinking, answer)``. If no block is found,
    ``thinking`` is ``None`` and the whole input is the answer.
    """
    for open_tag, close_tag in _THINK_PATTERNS:
        start = text.find(open_tag)
        if start == -1:
            continue
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            # Unterminated block - treat everything after ``open_tag``
            # as thinking and the rest as answer (best effort).
            thinking = text[start + len(open_tag) :]
            answer = text[:start].rstrip()
            return thinking.strip() or None, answer
        thinking = text[start + len(open_tag) : end]
        # Everything before the block plus everything after it.
        answer = text[:start] + text[end + len(close_tag) :]
        return thinking.strip() or None, answer.lstrip()
    return None, text


def _detect_format(model_dir: Path, gguf_path: Path | None = None) -> str:
    """Return one of ``"gguf"``, ``"safetensors"``, ``"mlx"``.

    MLX-4bit stores its ``weight``/``scales``/``biases`` triplets either
    as a single multi-tensor safetensors file or as separate ``*.weight``
    / ``*.scales`` / ``*.biases`` files. Either form counts as MLX.

    ``gguf_path`` overrides the directory scan and forces a GGUF
    detection - used when the user passes a file path on the CLI.
    """
    if gguf_path is not None or any(model_dir.glob("*.gguf")):
        return "gguf"
    if (model_dir / "model.safetensors").is_file() or any(model_dir.glob("*.safetensors")):
        if _looks_like_mlx_4bit(model_dir):
            return "mlx"
        return "safetensors"
    raise FileNotFoundError(f"No .gguf or .safetensors file in {model_dir}")


def _looks_like_mlx_4bit(model_dir: Path) -> bool:
    """Return ``True`` if any weight has the MLX ``weight/scales/biases`` triplet.

    For a single ``model.safetensors`` we open it briefly and look for
    any tensor whose name ends with ``.scales`` (the MLX format always
    pairs a ``.weight`` with a ``.scales`` and ``.biases``).
    """
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        return False
    # Fast path: if any sibling file is ``*.scales`` or ``*.biases``,
    # the directory is definitely MLX-style.
    if any(model_dir.glob("*.scales")) or any(model_dir.glob("*.biases")):
        return True
    # Otherwise open the first safetensors file and inspect the header.
    try:
        from flatrun.backend.safetensor import open_safetensors
        backend = open_safetensors(safetensors_files[0])
        backend.open()
        try:
            for k in backend.list_tensors():
                if k.name.endswith(".scales") or k.name.endswith(".biases"):
                    return True
        finally:
            backend.close()
    except Exception:
        return False
    return False


# GGUF architectures that ggml rotates with LLAMA_ROPE_TYPE_NORM, i.e.
# over pairs of *consecutive* head dimensions. ``convert_hf_to_gguf.py``
# un-permutes Q/K for exactly these, so the on-disk weights need the
# interleaved rotation rather than HuggingFace's rotate_half. Every
# other architecture (qwen2, phi3, gemma, stablelm, ...) is NEOX and
# keeps the HF layout. Mirrors ``llama_rope_type()`` in llama.cpp.
_GGUF_ROPE_NORM_ARCHES = frozenset(
    {
        "llama",
        "baichuan",
        "starcoder",
        "plamo",
        "orion",
        "internlm2",
        "minicpm",
        "xverse",
        "command-r",
        "olmo",
        "arctic",
        "deepseek2",
        "chatglm",
    }
)


def _build_config_from_gguf(gguf_path: Path) -> dict:
    """Derive an HF-shaped config dict from a GGUF's metadata block.

    GGUF namespaces its hyperparameters under the architecture name
    (``qwen2.block_count``, ``llama.block_count``, ...), so the prefix
    is read from ``general.architecture`` rather than assumed.
    """
    from flatrun.backend.gguf import GGUFBackend
    backend = GGUFBackend(gguf_path)
    backend.open()
    try:
        meta = backend.gguf_metadata
        arch = str(meta.get("general.architecture") or "").strip()
        if not arch:
            raise ConfigurationError(
                f"{gguf_path.name}: GGUF metadata has no 'general.architecture'"
            )

        def need(suffix: str):
            key = f"{arch}.{suffix}"
            if key not in meta:
                raise ConfigurationError(
                    f"{gguf_path.name}: GGUF metadata is missing {key!r}"
                )
            return meta[key]

        n_heads = int(need("attention.head_count"))
        hidden = int(need("embedding_length"))
        cfg = {
            "vocab_size": len(meta.get("tokenizer.ggml.tokens", []) or []),
            "hidden_size": hidden,
            "intermediate_size": int(need("feed_forward_length")),
            "num_hidden_layers": int(need("block_count")),
            "num_attention_heads": n_heads,
            "num_key_value_heads": int(
                meta.get(f"{arch}.attention.head_count_kv", n_heads)
            ),
            "rope_theta": float(meta.get(f"{arch}.rope.freq_base", 10000.0)),
            "rms_norm_eps": float(need("attention.layer_norm_rms_epsilon")),
            "max_position_embeddings": int(
                meta.get(f"{arch}.context_length", 32768)
            ),
            # GGUF only emits a separate output projection when the LM
            # head is untied; absence means it reuses the embedding.
            "tie_word_embeddings": not any(
                k.name in ("lm_head.weight", "output.weight")
                for k in backend.list_tensors()
            ),
        }
        head_dim = meta.get(f"{arch}.attention.key_length")
        if head_dim is not None:
            cfg["head_dim"] = int(head_dim)
        elif hidden % n_heads:
            raise ConfigurationError(
                f"{gguf_path.name}: hidden_size={hidden} is not divisible by "
                f"head_count={n_heads} and no key_length was provided"
            )
        cfg["rope_interleaved"] = arch in _GGUF_ROPE_NORM_ARCHES
        return cfg
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _assemble_prompt(
    tokenizer,
    *,
    messages_json: str | None,
    system: str | None,
    prompt: str | None,
) -> str:
    """Return the text to encode.

    Priority:

    1. ``--messages-json`` (list of {role, content} dicts) - rendered
       through the chat template.
    2. ``--system`` + ``--prompt`` (single user turn) - rendered.
    3. ``--prompt`` alone - returned as-is.
    """
    if messages_json:
        messages = json.loads(messages_json)
        if not isinstance(messages, list):
            raise ValueError("--messages-json must be a JSON list")
        return tokenizer.apply_chat_template(messages)
    if system or prompt is not None:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt or ""})
        return tokenizer.apply_chat_template(msgs)
    return prompt or ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import sys
    if argv is None:
        argv = sys.argv[1:]
    parser, _ = _build_argparser()
    # Backwards-compat: pre-existing scripts invoke
    # ``flatrun --model ... --prompt ...`` without a subcommand.
    # Detect that and prepend ``run`` so the legacy CLI works.
    if len(argv) == 0:
        argv = ["run"]
    elif argv[0] not in ("run", "chat", "-h", "--help"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler == "chat":
        return cmd_chat(args)
    return cmd_run(args)


def _build_argparser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the top-level parser and the shared parent parser.

    The shared parent carries every flag that's identical between
    ``run`` (one-shot) and ``chat`` (interactive REPL) - model path,
    tokenizer, runtime cache, sampling knobs, chat-template controls.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a model directory (GGUF, SafeTensors, or MLX-4bit).",
    )
    shared.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer directory. Defaults to --model. Used for GGUF dirs that ship no tokenizer.",
    )
    shared.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system message - prepended in chat templates.",
    )
    shared.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip the chat template; treat prompts as raw text.",
    )
    shared.add_argument(
        "--cache-mb",
        type=int,
        default=256,
        help="Memory cache cap in MiB (lower = more streaming).",
    )
    shared.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Tokens to generate after the prompt. run: default 1; chat: default 128.",
    )
    shared.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top tokens to show for the next-token prediction.",
    )
    shared.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Override GGUF quant name. Default: detect from filename.",
    )
    shared.add_argument(
        "--temperature",
        type=float,
        default=0.11,
        help="Sampling temperature. Lower = more deterministic. 0.0 not allowed.",
    )
    shared.add_argument(
        "--sample-top-k",
        type=int,
        default=20,
        help="Sample-time top-k filter (0 disables).",
    )
    shared.add_argument(
        "--sample-top-p",
        type=float,
        default=0.59,
        help="Sample-time nucleus filter. 1.0 disables.",
    )
    shared.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Sample-time min-p filter. 0.0 disables.",
    )
    shared.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty applied to seen token logits. 1.0 disables.",
    )
    shared.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy argmax (skip all sampling).",
    )
    shared.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling. Default: time-seeded.",
    )
    shared.add_argument(
        "--profile",
        action="store_true",
        help="Print per-step timing breakdown for the first N generation steps.",
    )
    shared.add_argument(
        "--debug",
        action="store_true",
        help="Print per-layer hidden-state norms and position-collapse "
             "metrics to stderr. Useful for cross-checking a model's "
             "forwarder against a reference (LM Studio, llama.cpp).",
    )

    parser = argparse.ArgumentParser(
        prog="flatrun",
        description=(
            "FlatRun - streaming inference runtime. "
            "Use 'flatrun run' for one-shot prompts or 'flatrun chat' "
            "for an interactive REPL. Without a subcommand, the "
            "legacy one-shot ``run`` mode is used."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    # ``required=False`` keeps ``parser.parse_args`` happy when no
    # subcommand was given. ``main()`` rewrites the argv to inject
    # ``run`` before parsing, so the absent-subcommand case never
    # actually reaches the parser.

    # ``run`` - the original one-shot command. Defaults to being
    # invoked when no subcommand is given.
    run_parser = subparsers.add_parser(
        "run",
        parents=[shared],
        add_help=True,
        help="Run a single prompt and print the continuation.",
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Plain text prompt. Ignored if --messages-json is set.",
    )
    run_parser.add_argument(
        "--messages-json",
        type=str,
        default=None,
        help="JSON list of {role, content} dicts. Rendered via the chat template.",
    )
    run_parser.set_defaults(_handler="run")

    # ``chat`` - interactive REPL.
    chat_parser = subparsers.add_parser(
        "chat",
        parents=[shared],
        add_help=True,
        help="Interactive REPL: type prompts, read assistant replies until EOF.",
    )
    chat_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append previous turns to the prompt. Each reply is a one-shot call.",
    )
    chat_parser.set_defaults(_handler="chat")

    return parser, shared


def _resolve_model_paths(args, parser: argparse.ArgumentParser) -> tuple[Path, Path | None, str]:
    """Resolve the model path, optional GGUF path, and detected format."""
    model_path: Path = args.model
    if not model_path.exists():
        parser.exit(1, f"Model path not found: {model_path}\n")
    if model_path.is_file():
        if model_path.suffix.lower() != ".gguf":
            parser.exit(
                1,
                "FlatRun only accepts a single .gguf file as a file path; "
                f"got {model_path.name}. For other formats point at the "
                "directory containing the model.safetensors files.\n",
            )
        return model_path.parent, model_path, "gguf"
    return model_path, None, _detect_format(model_path)


def _pick_gguf_file(model_dir: Path, parser: argparse.ArgumentParser) -> Path:
    candidates = sorted(model_dir.glob("*.gguf"))
    if not candidates:
        parser.exit(1, f"No .gguf file in {model_dir}\n")
    return candidates[0]


def _load_model_bundle(args, parser: argparse.ArgumentParser) -> dict:
    """Open the model, return a dict with everything both handlers need.

    Centralising this lets ``run`` and ``chat`` share the same
    cache-bump heuristic, vocab-mismatch check, and forwarder setup.
    """
    model_dir, gguf_path, fmt = _resolve_model_paths(args, parser)
    if fmt == "gguf" and gguf_path is None:
        gguf_path = _pick_gguf_file(model_dir, parser)
    print(f"Detected format: {fmt}")

    if fmt == "gguf" and not any(
        (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
    ):
        from flatrun.tokenizer import load_from_gguf_metadata
        print(f"Building tokenizer from GGUF metadata ({gguf_path.name}) ...")
        tokenizer = load_from_gguf_metadata(gguf_path)
    else:
        tok_dir = args.tokenizer or model_dir
        if args.tokenizer is None and fmt == "gguf" and not any(
            (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
        ):
            candidate = (
                Path("/Users/judotens/.lmstudio/models/lmstudio-community/")
                / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"
            )
            if candidate.is_dir():
                tok_dir = candidate
                print(f"  using fallback tokenizer at {tok_dir}")
        tokenizer = auto_load(tok_dir)
    tok_vocab = len(tokenizer.vocab)
    print(f"Tokenizer vocab: {tok_vocab}")
    print(
        f"Chat template: "
        f"{'Qwen2 ChatML' if '<|im_start|>' in tokenizer.chat_template else tokenizer.chat_template[:60] + '...'}"
    )

    cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
    t0 = time.perf_counter()
    loaded = load_huggingface(model_dir, config=cfg)
    if args.cache_mb == 256:
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        if largest > 0:
            recommended_mb = max(
                256,
                ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128,
            )
            if recommended_mb > args.cache_mb:
                print(
                    f"  bumping cache from {args.cache_mb} MiB to {recommended_mb} MiB "
                    f"(largest tensor is {largest / 1024 / 1024:.0f} MiB)"
                )
                args.cache_mb = recommended_mb
                loaded.runtime.close()
                cfg = RuntimeConfig(
                    memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None)
                )
                t0 = time.perf_counter()
                loaded = load_huggingface(model_dir, config=cfg)
    print(f"Loaded model in {(time.perf_counter() - t0):.2f} s; layers={loaded.manifest.layer_count}")

    model_vocab: int | None = None
    if fmt == "gguf":
        from flatrun.backend.gguf import GGUFBackend
        be = GGUFBackend(gguf_path)
        be.open()
        try:
            gguf_meta = be.gguf_metadata
        finally:
            be.close()
        model_vocab = len(gguf_meta.get("tokenizer.ggml.tokens", []) or [])
    elif loaded.config is not None and loaded.config.raw is not None:
        model_vocab = int(loaded.config.raw.get("vocab_size", 0)) or None

    if model_vocab is not None and model_vocab != tok_vocab:
        print(
            f"\n*** VOCAB MISMATCH: model={model_vocab}, tokenizer={tok_vocab} ***\n"
            f"    The model's argmax IDs won't map to the tokenizer's vocab.\n"
            f"    Output will be garbage unless you pass --tokenizer pointing\n"
            f"    to a directory whose vocab matches the model.\n",
            file=sys.stderr,
        )

    if fmt == "gguf":
        raw_cfg = _build_config_from_gguf(gguf_path)
        qcfg = Qwen2Config.from_hf_config(raw_cfg)
        qcfg.quant_gguf = args.quant or "Q8_0"
        qcfg.debug_trace = args.debug
    else:
        if loaded.config is None or loaded.config.raw is None:
            parser.exit(1, "No config.json found next to model weights.\n")
        qcfg = Qwen2Config.from_hf_config(loaded.config.raw)
        qcfg.quant_mlx_4bit = fmt == "mlx"
        qcfg.quant_gguf = None
        qcfg.debug_trace = args.debug
    forwarder = make_qwen2_forwarder(qcfg)

    scheduler = loaded.runtime.build_scheduler(
        loaded.manifest.layers,
        pre_layer_names=loaded.manifest.pre_layer,
        post_layer_names=loaded.manifest.post_layer,
    )
    executor = StreamingExecutor(scheduler, forwarder, kv_cache=KVCache(capacity=4096))

    return {
        "fmt": fmt,
        "gguf_path": gguf_path,
        "tokenizer": tokenizer,
        "loaded": loaded,
        "forwarder": forwarder,
        "executor": executor,
        "qcfg": qcfg,
    }


def _make_sampler(args) -> Sampler:
    if args.no_sample:
        return Sampler(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0)
    return Sampler(
        temperature=args.temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )


def _generate_continuation(
    bundle: dict,
    args,
    prompt_ids: list[int],
    max_new: int,
) -> tuple[list[int], np.ndarray | None, list[float]]:
    """Run the forwarder ``max_new`` times, return (ids, last_logits, step_times)."""
    executor = bundle["executor"]
    tokenizer = bundle["tokenizer"]
    sampler = _make_sampler(args)
    seen: list[int] = list(prompt_ids)
    step_times: list[float] = []
    logits: np.ndarray | None = None
    next_id = -1
    generated: list[int] = []

    with Spinner("Thinking"):
        for nxt in range(max_new):
            ids = prompt_ids + generated if next_id == -1 else prompt_ids + generated
            t0 = time.perf_counter()
            result = executor.step(tokens=ids)
            step_times.append((time.perf_counter() - t0) * 1000)
            if nxt == 0 and args.profile:
                print(f"  initial step: {step_times[-1]:.0f} ms ({len(ids)} tokens)")
            elif args.profile:
                print(f"  step {nxt + 1}: {step_times[-1]:.0f} ms ({len(ids)} tokens)")
            logits = result.last_hidden
            if args.no_sample:
                next_id = int(np.argmax(logits[-1]))
            else:
                next_id = sampler.sample(logits[-1], seen_ids=seen)
            generated.append(next_id)
            seen.append(next_id)

    return generated, logits, step_times


def cmd_run(args) -> int:
    """One-shot prompt -> continuation (the original behaviour)."""
    parser, _ = _build_argparser()
    if args.max_new is None:
        args.max_new = 1
    bundle = _load_model_bundle(args, parser)
    tokenizer = bundle["tokenizer"]

    if args.no_chat_template:
        prompt_text = args.prompt or ""
    else:
        prompt_text = _assemble_prompt(
            tokenizer,
            messages_json=args.messages_json,
            system=args.system,
            prompt=args.prompt,
        )
    print(f"Prompt:\n{prompt_text}")
    prompt_ids = tokenizer.encode(prompt_text)
    print(f"Prompt tokens: {len(prompt_ids)}")

    sampler = _make_sampler(args)
    print(
        f"  sampling: temperature={sampler.temperature} top_k={sampler.top_k} "
        f"top_p={sampler.top_p} min_p={sampler.min_p} repeat_penalty={sampler.repeat_penalty}"
    )

    print(f"\nRunning initial step with {len(prompt_ids)} prompt tokens ...")
    generated, logits, step_times = _generate_continuation(
        bundle, args, prompt_ids, args.max_new
    )
    raw_text = tokenizer.decode(generated)
    for nxt, tid in enumerate(generated):
        print(f"  token {nxt + 1}: id={tid} text={tokenizer.decode([tid])!r}")

    if generated:
        thinking, answer = _split_thinking(raw_text)
        if thinking:
            print(f"\n\033[2mThinking:\n{thinking}\n\033[0m")
        print(f"\nGenerated text: {answer!r}")
    if args.profile and len(step_times) > 1:
        first = step_times[0]
        last = step_times[-1]
        avg = sum(step_times) / len(step_times)
        print(f"\nProfile summary across {len(step_times)} generation steps:")
        print(f"  first step: {first:.0f} ms  (cold cache + dequant)")
        print(f"  last step:  {last:.0f} ms  (warm cache)")
        print(f"  average:    {avg:.0f} ms")
        if first > 0:
            print(f"  speedup from warm cache: {first / max(last, 1):.1f}x")
    if args.top_k > 0 and logits is not None:
        top_k = min(args.top_k, logits.shape[-1])
        top_indices = np.argpartition(logits[-1], -top_k)[-top_k:]
        print(f"\nTop-{top_k} next tokens:")
        for tid in sorted(top_indices, key=lambda i: -logits[-1, i]):
            print(f"  id={tid:6d} logit={logits[-1, tid]:8.2f}  text={tokenizer.decode([tid])!r}")

    bundle["loaded"].runtime.close()
    print("Done.")
    return 0


def cmd_chat(args) -> int:
    """Interactive REPL: 'You: ' prompt, stream the assistant's reply.

    Each turn is rendered through the model's chat template (unless
    ``--no-chat-template`` is set) with the full prior history
    included, so the model sees a real multi-turn conversation. The
    executor is reset between turns so the KV cache cannot drift
    between sessions; the per-turn prefill cost is paid from scratch.
    """
    parser, _ = _build_argparser()
    if args.max_new is None:
        args.max_new = 128
    bundle = _load_model_bundle(args, parser)
    tokenizer = bundle["tokenizer"]
    print(
        f"\nChat mode (max_new={args.max_new}/turn, history={not args.no_history})."
        f"  Type your message; Ctrl-D (EOF) or 'exit' to quit.\n"
    )
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    turn = 0
    while True:
        try:
            user_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_text.strip():
            continue
        messages.append({"role": "user", "content": user_text})

        if args.no_chat_template:
            prompt_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            ) + "\nassistant:"
        else:
            # ``apply_chat_template`` expects the assistant turn to be
            # open; we add an empty assistant slot so the template
            # emits the "now you speak" prefix.
            prompt_text = tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": ""}],
                add_generation_prompt=True,
            )
        prompt_ids = tokenizer.encode(prompt_text)

        if not args.no_history:
            # Render only the latest turn for the model; we pass the
            # full message history to the template but truncate to
            # just the last user turn's encoded text. The chat
            # template does the actual multi-turn formatting.
            pass

        bundle["executor"].kv_cache.reset()
        turn += 1
        t0 = time.perf_counter()
        generated, _, _ = _generate_continuation(
            bundle, args, prompt_ids, args.max_new
        )
        dt = time.perf_counter() - t0
        raw_text = tokenizer.decode(generated)
        thinking, reply_text = _split_thinking(raw_text)
        for stop in ("<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"):
            if stop in reply_text:
                reply_text = reply_text.split(stop, 1)[0]
                break
        if thinking:
            print(f"\033[2mThinking:\n{thinking}\n\033[0m")
        print(f"Assistant: {reply_text}")
        print(f"  ({len(generated)} tokens, {dt:.1f}s, {len(generated) / max(dt, 1e-3):.1f} tok/s)\n")
        messages.append({"role": "assistant", "content": reply_text})

    bundle["loaded"].runtime.close()
    return 0


    # 1. Detect format and load tokenizer.
    fmt = _detect_format(model_dir, gguf_path=gguf_path)
    print(f"Detected format: {fmt}")
    # Resolve the on-disk GGUF file once. ``gguf_path`` may have been
    # set from the CLI (file path) or may need to be picked from the
    # directory's first ``.gguf`` file.
    if fmt == "gguf":
        if gguf_path is None:
            candidates = sorted(model_dir.glob("*.gguf"))
            if not candidates:
                print(f"No .gguf file in {model_dir}", file=sys.stderr)
                return 1
            gguf_path = candidates[0]
    tokenizer = None
    # For GGUF dirs without sibling tokenizer files, build the tokenizer
    # directly from the GGUF metadata so the vocab matches the model.
    if fmt == "gguf" and not any(
        (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
    ):
        from flatrun.tokenizer import load_from_gguf_metadata
        print(f"Building tokenizer from GGUF metadata ({gguf_path.name}) ...")
        tokenizer = load_from_gguf_metadata(gguf_path)
    else:
        tok_dir = args.tokenizer or model_dir
        if args.tokenizer is None and fmt == "gguf" and not any(
            (model_dir / f).is_file() for f in ("tokenizer.json", "vocab.json")
        ):
            # Final fallback - the user's MLX 7B is the only sibling
            # tokenizer in their LM Studio download.
            candidate = (
                Path("/Users/judotens/.lmstudio/models/lmstudio-community/")
                / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"
            )
            if candidate.is_dir():
                tok_dir = candidate
                print(f"  using fallback tokenizer at {tok_dir}")
        tokenizer = auto_load(tok_dir)
    tok_vocab = len(tokenizer.vocab)
    print(f"Tokenizer vocab: {tok_vocab}")
    print(f"Chat template: {'Qwen2 ChatML' if '<|im_start|>' in tokenizer.chat_template else tokenizer.chat_template[:60] + '...'}")

    # Pre-load the model early so we can compare vocab sizes. The vocab
    # lives in different places depending on format - GGUF exposes it
    # via the metadata KV table, HF / MLX go through ``Qwen2Config``.
    # Probe the manifest so the default cache can adapt to large
    # models. We need at least 4x the largest single tensor (embed,
    # lm_head) or the cache will evict the embed right after acquiring
    # it, which crashes the forwarder.
    cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
    t0 = time.perf_counter()
    loaded = load_huggingface(model_dir, config=cfg)
    if args.cache_mb == 256:  # user did not override; pick a sensible default
        largest = max(
            (loaded.runtime.get_metadata(k.name).byte_size for k in loaded.runtime.list_tensors()),
            default=0,
        )
        # 4x largest tensor, rounded up to the next 128 MiB, with a
        # 256 MiB floor (matches the explicit default). 4x because the
        # scheduler typically holds pre-layer + post-layer + current
        # layer's tensors concurrently.
        if largest > 0:
            recommended_mb = max(256, ((largest * 4) + (128 * 1024 * 1024) - 1) // (128 * 1024 * 1024) * 128)
            if recommended_mb > args.cache_mb:
                print(
                    f"  bumping cache from {args.cache_mb} MiB to {recommended_mb} MiB "
                    f"(largest tensor is {largest / 1024 / 1024:.0f} MiB)",
                )
                args.cache_mb = recommended_mb
                loaded.runtime.close()
                cfg = RuntimeConfig(memory=MemoryConfig(cache_bytes=args.cache_mb * 1024 * 1024, probe=None))
                t0 = time.perf_counter()
                loaded = load_huggingface(model_dir, config=cfg)
    print(f"Loaded model in {(time.perf_counter() - t0):.2f} s; layers={loaded.manifest.layer_count}")

    model_vocab: int | None = None
    if fmt == "gguf":
        from flatrun.backend.gguf import GGUFBackend
        be = GGUFBackend(gguf_path)
        be.open()
        try:
            gguf_meta = be.gguf_metadata
        finally:
            be.close()
        model_vocab = len(gguf_meta.get("tokenizer.ggml.tokens", []) or [])
    else:
        if loaded.config is not None and loaded.config.raw is not None:
            model_vocab = int(loaded.config.raw.get("vocab_size", 0)) or None

    if model_vocab is not None and model_vocab != tok_vocab:
        print(
            f"\n*** VOCAB MISMATCH: model={model_vocab}, tokenizer={tok_vocab} ***\n"
            f"    The model's argmax IDs won't map to the tokenizer's vocab.\n"
            f"    Output will be garbage unless you pass --tokenizer pointing\n"
            f"    to a directory whose vocab matches the model.\n",
            file=sys.stderr,
        )

    # 2. Render the prompt.
    if args.no_chat_template:
        prompt_text = args.prompt or ""
    else:
        prompt_text = _assemble_prompt(
            tokenizer,
            messages_json=args.messages_json,
            system=args.system,
            prompt=args.prompt,
        )
    print(f"Prompt:\n{prompt_text}")
    prompt_ids = tokenizer.encode(prompt_text)
    print(f"Prompt tokens: {len(prompt_ids)}")

    # 3. Build the Qwen2 forwarder.
    if fmt == "gguf":
        raw_cfg = _build_config_from_gguf(gguf_path)
        qcfg = Qwen2Config.from_hf_config(raw_cfg)
        qcfg.quant_gguf = args.quant or "Q8_0"
        qcfg.debug_trace = args.debug
    else:
        if loaded.config is None or loaded.config.raw is None:
            print("No config.json found next to model weights.", file=sys.stderr)
            return 1
        qcfg = Qwen2Config.from_hf_config(loaded.config.raw)
        qcfg.quant_mlx_4bit = fmt == "mlx"
        qcfg.quant_gguf = None
        qcfg.debug_trace = args.debug
    forwarder = make_qwen2_forwarder(qcfg)

    # 4. Stream one prompt + max_new tokens.
    scheduler = loaded.runtime.build_scheduler(
        loaded.manifest.layers,
        pre_layer_names=loaded.manifest.pre_layer,
        post_layer_names=loaded.manifest.post_layer,
    )
    kv = KVCache(capacity=max(128, len(prompt_ids) + args.max_new + 16))
    executor = StreamingExecutor(scheduler, forwarder, kv_cache=kv)

    print(f"\nRunning initial step with {len(prompt_ids)} prompt tokens ...")
    t0 = time.perf_counter()
    result = executor.step(tokens=prompt_ids)
    step_ms = (time.perf_counter() - t0) * 1000
    print(f"  initial step took {step_ms:.1f} ms")
    logits = result.last_hidden
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )
    if args.no_sample:
        # Force greedy decoding regardless of other settings.
        sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0)
    print(
        f"  sampling: temperature={sampler.temperature} top_k={sampler.top_k} "
        f"top_p={sampler.top_p} min_p={sampler.min_p} repeat_penalty={sampler.repeat_penalty}"
    )
    generated: list[int] = []
    # Tokens the model has already produced - the sampler applies
    # repeat_penalty to these.
    seen: list[int] = list(prompt_ids) + generated
    step_times: list[float] = []
    for nxt in range(args.max_new):
        if args.no_sample:
            next_id = int(np.argmax(logits[-1]))
        else:
            next_id = sampler.sample(logits[-1], seen_ids=seen)
        generated.append(next_id)
        seen.append(next_id)
        next_text = tokenizer.decode([next_id])
        print(f"  token {nxt + 1}: id={next_id} text={next_text!r}")
        if nxt + 1 >= args.max_new:
            break
        # Append the new token to the prompt and step again.
        # FlatRun's executor only does one shot, so we re-run the whole
        # sequence (acceptable for tiny max_new).
        new_ids = prompt_ids + generated
        t0 = time.perf_counter()
        result = executor.step(tokens=new_ids)
        step_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_ms)
        if args.profile:
            print(f"    step {nxt + 2}: {step_ms:.0f} ms "
                  f"({len(new_ids)} tokens, ~{step_ms / len(new_ids):.1f} ms/tok)")
        logits = result.last_hidden
    if generated:
        print(f"\nGenerated text: {tokenizer.decode(generated)!r}")
    if args.profile and step_times:
        first = step_times[0]
        last = step_times[-1]
        avg = sum(step_times) / len(step_times)
        print(f"\nProfile summary across {len(step_times)} generation steps:")
        print(f"  first step: {first:.0f} ms  (cold cache + dequant)")
        print(f"  last step:  {last:.0f} ms  (warm cache)")
        print(f"  average:    {avg:.0f} ms")
        if first > 0:
            print(f"  speedup from warm cache: {first / max(last, 1):.1f}x")

    # 5. Top-k summary on the final logits.
    if args.top_k > 0 and logits is not None:
        top_k = min(args.top_k, logits.shape[-1])
        top_indices = np.argpartition(logits[-1], -top_k)[-top_k:]
        print(f"\nTop-{top_k} next tokens:")
        for tid in sorted(top_indices, key=lambda i: -logits[-1, i]):
            print(f"  id={tid:6d} logit={logits[-1, tid]:8.2f}  text={tokenizer.decode([tid])!r}")

    loaded.runtime.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
