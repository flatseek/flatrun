# Model matrix

This is the live set of models FlatRun 0.1.0 has been tested
against. Each row is the model we actually drove through the CLI
in greedy mode (`--no-sample`) and compared to LM Studio
(`temperature=0`, `top_k=1`, `repeat_penalty=1.0`).

A ✅ in the *LM Studio match* column means the first 12 generated
tokens are byte-identical to the server response. A ⚠️ means the
output is fluent English and on-topic but diverges on later
tokens due to numerical drift between FlatRun's float32 matmul
and llama.cpp's Q8_1-quantised activations.

## SafeTensors (HuggingFace layout)

| Model | File | Dtype | LM Studio match | Notes |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | `model.safetensors` | bf16 | ✅ | GQA (9/3) |
| Qwen2.5-0.5B | `model.safetensors` | bf16 | ✅ | dense |

## GGUF (Q4_K_M, Q5_K_M, Q6_K, Q8_0)

| Model | File | LM Studio match | Notes |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | `Qwen2.5-Coder-0.5B-Q8_0.gguf` | ✅ | NEOX RoPE |
| SmolLM2-360M-Instruct | `smollm2-360m-instruct-q8_0.gguf` | ✅ | NORM RoPE |
| Qwen3-0.6B | `Qwen3-0.6B-Q4_K_M.gguf` | ✅ | per-head q/k norm |
| Bonsai-1.7B | `Bonsai-1.7B-Q1_0.gguf` | ✅ | 1-bit, 1.125 bpw |
| Qwen3.5-0.8B MLX-4bit | `Qwen3.5-0.8B-MLX-4bit` | ⚠️ partial | Architecture decoded, full-attention path implemented; **DeltaNet linear-attention raises NotImplementedError** on first linear layer. |

## Requantised (K-quants produced locally for decoder testing)

| Source | Output quant | Decoder test |
|---|---|---|
| SmolLM2-360M Q8_0 | Q4_K_M | ✅ koheren |
| SmolLM2-360M Q8_0 | Q6_K | ✅ koheren |

## Not yet supported

These are on the 0.2.0 list. They currently produce garbage or
crash on load; we list them so a future maintainer can pick
where to start.

| Model | Why |
|---|---|
| Qwen3.5 / Qwen3.5-MoE | hybrid linear + full attention (DeltaNet), per-layer type selection, `attn_output_gate` |
| Gemma 3 1B MLX-4bit | ⚠️ partial | Architecture decoded; **logits still flat** (see note) |
| Phi-3 | fused QKV / gate_up matmul, different RoPE scale |
| Gemma 2 | different norm placement, logit soft-capping |
| Mistral | sliding-window attention not implemented |

Adding a new architecture means (a) extending
`flatrun.model.qwen2._decoder_block` with the new op, (b) the
chat-template fallback in `flatrun.tokenizer.bpe`, and (c) a
couple of unit tests built around a tiny synthetic model.
