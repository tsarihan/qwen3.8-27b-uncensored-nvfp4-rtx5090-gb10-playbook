---
license: apache-2.0
base_model: JonathanColetti/Qwen3.8-27B-Uncensored
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - nvfp4
  - fp4
  - compressed-tensors
  - vllm
  - blackwell
  - rtx-5090
  - dgx-spark
  - gb10
  - mtp
  - speculative-decoding
  - uncensored
  - abliterated
  - ai-red-team
  - red-teaming
  - vision-language
  - quantized
extra_gated_prompt: >-
  **This model has had its refusal behaviour removed.**


  It is a quantization of an abliterated checkpoint. Per the upstream author's own
  measurements, refusal on harmful prompts drops from 98/100 to 12/100. It will comply
  with requests that the original Qwen3.8-27B would decline, and it has no meaningful
  built-in guardrails.


  It is released for authorized security testing, red-team evaluation, refusal-mechanism
  research and alignment research in isolated environments. It is **not** a general-purpose
  assistant. For any other purpose use
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), to which this quantization
  recipe applies unchanged.


  Do not deploy this model to end users or to any public endpoint without your own
  independent safety, moderation and abuse-prevention layer. You assume full
  responsibility and liability for what you do with it and for what it produces.
extra_gated_fields:
  I understand this model's refusal behaviour has been removed and it has no built-in guardrails: checkbox
  I will use it only for authorized research or testing in an isolated environment: checkbox
  I will not deploy it to end users or a public endpoint without my own safety and moderation layer: checkbox
  I accept full responsibility and liability for its outputs: checkbox
extra_gated_button_content: I acknowledge the risks and agree
---

# Qwen3.8-27B-Uncensored-NVFP4

NVFP4 + FP8 mixed-precision quantization of
[`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored),
for Blackwell FP4 tensor cores. **23.42 GB**, fits a 32 GB RTX 5090 with room for KV, and
runs the preserved **MTP draft head** for speculative decoding.

Vision tower and MTP head are kept in BF16. Context 262,144 native.

> Read the gated notice above before using this. This checkpoint has no guardrails.

## Why this exists

The upstream author publishes bf16 and GGUF but **no NVFP4**. The only NVFP4 derived from
those weights is a third-party requant by someone other than the author of the weight edit.
This is a conversion of the author's own bf16, sha256-verified against their LFS hashes
(`VERIFY_OK`, 14/14) before conversion.

**No retraining, no fine-tuning, no GPTQ, no AWQ.** Weights are converted by
round-to-nearest into the FP4/FP8 grids. Calibration is forward-passes-only and never
alters a weight — it exists solely to observe activation ranges for the FP4 activation
global scales and the static FP8 KV-cache scales, neither of which can be derived from
weights. 512 samples of wikitext-2 at sequence length 2048. No instruction or task data.

## Accuracy

Perplexity on wikitext-2, 24 paired chunks of 2048 tokens (49,128 tokens), measured
**through the live vLLM serving path** so the actual FP4/FP8 kernels are in the loop:

| model | PPL | vs bf16 parent |
|---|---|---|
| bf16 parent | 6.8387 | baseline |
| **this checkpoint (NVFP4 W4A4)** | **6.9261** | **+0.0874 (+1.28%)** |
| NVFP4A16 weight-only (not published) | 7.2574 | +0.4187 (+6.12%) |

For scale, the abliteration itself costs +0.0434 PPL over base Qwen3.8-27B by the upstream
author's measurement, so this conversion costs roughly double the weight edit.

The third row is included as a warning: a **weight-only** NVFP4 conversion of the same
weights was 4.8× worse, and scored **20/20 needles out to 248,419 tokens** regardless.
Needle retrieval does not detect this damage class.

## Quantization

`format: mixed-precision`, two config groups. Only layers 0-55 MLP get 4 bits.

| component | precision |
|---|---|
| MLP gate/up/down, layers 0-55 | **NVFP4** W4A4, group_size 16 |
| MLP layers 56-63 | FP8 (sensitivity carve-out) |
| self_attn q/k/v/o | FP8 W8A8, per-channel |
| `linear_attn` in_proj_qkv / in_proj_z / out_proj | FP8 |
| `linear_attn` in_proj_a / in_proj_b | **not quantized** |
| lm_head | FP8 |
| **MTP head** (15 tensors) | **BF16**, `re:^mtp.*` ignored |
| **vision tower** (333 tensors) | **BF16**, ignored |
| KV cache | static calibrated **FP8** |

168 packed FP4 / 400 FP8 scales / 168 global scales / 585 BF16, 303 ignore entries.

## Serving

Requires a Blackwell GPU for native FP4. **Do not pass `--quantization` or
`--kv-cache-dtype`** — `config.json` carries the mixed-precision groups *and* the
calibrated static FP8 `kv_cache_scheme`, and those flags override them.

```bash
docker run -d --gpus all --ipc=host --network host --shm-size=16g \
  -v /path/to/models:/models \
  vllm/vllm-openai:v0.27.1 \
  --model /models/Qwen3.8-27B-Uncensored-NVFP4 \
  --served-model-name qwen3.8-27b-uncensored-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 98304 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.945 --enable-prefix-caching --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

vLLM **0.28 does not run on WSL2** (`RuntimeError: UVA is not available`); use 0.27.x there.

## Speculative decoding (MTP)

RTX 5090, vLLM v0.27.1, ctx 98304, acceptance 34.1%:

| streams | no MTP | **k=3** | per-stream gain |
|---|---|---|---|
| 1 | 61.34 | **80.71** | **+31.6%** |
| 2 | 55.71 | **77.61** | **+39.3%** |
| 4 | 54.86 | **71.80** | **+30.9%** |
| 8 | 56.31 | 71.50 | +27.0% |
| 64 | 52.77 | 66.38 | +25.8% |

Aggregate throughput also wins up to 4 streams (79.6 / 145.0 / 230.3 vs 61.0 / 109.7 /
213.7) and inverts past 8. **Use k=3 at ≤4 streams; disable MTP beyond 8** if you need
aggregate throughput.

Prefill does not collapse: 5,982 / 7,707 / 6,094 tok/s at 4K/16K/64K under k=3.

**Depth does not transfer between GPUs.** On a DGX Spark GB10 the same family peaks at
**k=1** and falls below the no-MTP baseline by k=5, because it is bandwidth-bound and
lacks compute for deep draft chains. Measure `num_speculative_tokens` on your own hardware.

## Reproducing

Full recipe, scripts, benchmarks and the failures along the way:
**https://github.com/tsarihan/qwen3.8-27b-uncensored-nvfp4-rtx5090-gb10-playbook**

## Attribution and license

Apache-2.0, matching upstream.

* Abliteration and bf16 weights: [JonathanColetti](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored),
  using [Heretic](https://github.com/p-e-w/heretic)
* Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
* Quantization: `llm-compressor` / `compressed-tensors`

Benchmarks were measured on the author's own hardware and are reported as measured.
Quant-specific accuracy beyond the perplexity table above has not been evaluated; there is
no SWE-bench Pro or capability benchmark for this checkpoint yet.
