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


  It is a quantization of an abliterated checkpoint. On the upstream author's measurement,
  refusal on harmful prompts falls from 98/100 to 12/100. It will attempt requests the
  original Qwen3.8-27B declines, and it has no meaningful built-in guardrails. Refusals are
  reduced, not eliminated, and behaviour near the old refusal boundary is less stable than
  the base model's.


  It is released for authorized security testing, red-team evaluation, refusal-mechanism
  research and alignment research in isolated environments. It is **not** a general-purpose
  assistant. For any other purpose use
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — this quantization recipe
  applies to it unchanged.


  Do not deploy this model to end users or to any public endpoint without your own
  independent safety, moderation and abuse-prevention layer. You assume full responsibility
  and liability for how you use it and for what it produces. The base model's license and
  acceptable use policy still apply to your use of this derivative.
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
built for **Blackwell FP4 tensor cores**.

**23.42 GB** — fits a 32 GB RTX 5090 with room for KV cache, and runs comfortably on a
DGX Spark GB10. The preserved **MTP draft head** works for speculative decoding, the
**vision tower is intact**, and native context is **262,144**.

> Read the gated notice above before using this. This checkpoint has no guardrails.

## Highlights

- **Native FP4.** W4A4 on the MLP means vLLM selects `FlashInferCutlassNvFp4LinearKernel`
  rather than the Marlin weight-only fallback. Verified on both SM120 and sm_121.
- **+31.6% single-stream** with MTP `num_speculative_tokens=3` on an RTX 5090.
- **+1.28% perplexity** against the bf16 parent — measured, not assumed.
- **MTP head preserved** (15 tensors, BF16) and **vision tower preserved** (333 tensors,
  BF16), both declared unquantized so vLLM loads them correctly.
- **Static calibrated FP8 KV cache** declared in `config.json`, so no
  `--kv-cache-dtype` flag is needed or wanted.

## Model overview

| | |
|---|---|
| Base | `Qwen/Qwen3.8-27B` |
| Abliteration | `JonathanColetti/Qwen3.8-27B-Uncensored` (Heretic) |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Layers | 64 — `16 × (3 × Gated DeltaNet → FFN, then 1 × Gated Attention → FFN)` |
| Hidden / FFN | 5120 / 17408 |
| Vocab | 248,320 |
| Context | 262,144 native, YaRN-extensible to 1M |
| Vision | yes, BF16, preserved |
| MTP | 1 layer, BF16, preserved |
| Size | 23.42 GB, 1,968 tensors, 3 shards |

## Quantization

`format: mixed-precision`, two config groups. **Only layers 0-55 MLP get 4 bits.**

| component | precision |
|---|---|
| MLP gate/up/down, layers 0-55 | **NVFP4** W4A4, group_size 16 |
| MLP gate/up/down, layers 56-63 | FP8 W8A8 — sensitivity carve-out |
| `self_attn` q/k/v/o | FP8 W8A8, per-channel |
| `linear_attn` in_proj_qkv / in_proj_z / out_proj | FP8 W8A8 |
| `linear_attn` in_proj_a / in_proj_b | **not quantized** |
| lm_head | FP8 |
| MTP head (15 tensors) | **BF16**, `re:^mtp.*` ignored |
| Vision tower (333 tensors) | **BF16**, ignored |
| KV cache | static calibrated **FP8** |

168 packed FP4 / 400 FP8 scales / 168 global scales / 585 BF16 · 303 ignore entries.

**No retraining, no fine-tuning, no GPTQ, no AWQ.** Weights convert by round-to-nearest
into the FP4/FP8 grids. Calibration is forward-passes-only and never alters a weight — it
exists solely to observe activation ranges for the FP4 activation global scales and the
static FP8 KV-cache scales, neither of which can be derived from weights. 512 samples of
wikitext-2 at sequence length 2048. No instruction data, no task data.

### Why the carve-outs matter

A **weight-only** (`NVFP4A16`) conversion of these same weights, quantizing all 496
language-model linears, measured **+0.4187 PPL (+6.12%)** — roughly 4.8× worse than this
build. Quantizing attention and the Gated DeltaNet projections to 4 bits is what costs
accuracy; FP4 *activations* on the MLP cost very little. Weight-only also cannot use the
FP4 tensor cores, because a native FP4 GEMM needs both operands in FP4.

## Evaluation

### Perplexity

wikitext-2, 24 paired chunks of 2048 tokens (49,128 tokens), measured **through the live
vLLM serving path** so the actual FP4/FP8 kernels are in the loop.

| model | PPL | vs bf16 parent |
|---|---|---|
| bf16 parent | 6.8387 | baseline |
| **this checkpoint** | **6.9261** | **+0.0874 (+1.28%)** |
| weight-only NVFP4A16 (not published) | 7.2574 | +0.4187 (+6.12%) |

For scale, the abliteration itself costs +0.0434 PPL over base Qwen3.8-27B on the upstream
author's measurement — so this conversion costs about double the weight edit.

### Long-context retrieval

5 needles at depths 0.10 / 0.30 / 0.50 / 0.70 / 0.90, distinct nonce per run, token counts
verified against the server's `/tokenize`. DGX Spark GB10, ctx 262144, no speculation.

| target | prompt tokens | TTFT | prefill tok/s | decode tok/s | needles |
|---|---|---|---|---|---|
| 4,096 | 4,315 | 1.76 s | 2,449.9 | 11.23 | **5/5** |
| 32,768 | 33,370 | 15.94 s | 2,093.3 | 10.75 | **5/5** |
| 131,072 | 132,970 | 109.49 s | 1,214.4 | 9.34 | **5/5** |
| 245,000 | 248,416 | 305.53 s | 813.1 | 8.12 | **5/5** |

**20/20 needles at every depth. Max context with all needles retrieved: 248,416.**

### Vision

3/3 on rendered probes — printed digits, coloured-shape counting, and relative magnitude
in a bar chart. The tower is BF16 and untouched by quantization.

### How to read these

- **Perplexity detects gross quantization damage and little else.** It does not measure
  reasoning, code, multilingual ability, or refusal behaviour.
- **Needle retrieval is weaker still for this purpose.** The weight-only build above scored
  the same 20/20 out to 248,419 tokens while being 6% worse in perplexity. If you validate
  a quantization on NIAH alone you will ship a degraded checkpoint and call it verified.
- **Rows are paired**, over identical chunks, so they are comparable to each other — but
  not to numbers computed over a different amount of text, including the upstream author's
  own tables.
- **Nothing here is a capability benchmark.** No MMLU, no GSM8K, no HumanEval, no
  SWE-bench. The upstream abliteration reports a 0.5-point mean drop vs base across MMLU,
  ARC-Challenge, HellaSwag and Winogrande; quantization compounds whatever that is, and
  this checkpoint has not been measured on any of them.

## Deployment

Requires a Blackwell GPU for native FP4. **Do not pass `--quantization` or
`--kv-cache-dtype`** — `config.json` carries the mixed-precision `config_groups` *and* the
calibrated static FP8 `kv_cache_scheme`, and those flags override them.

### RTX 5090 (SM120, 32 GB)

```bash
docker run -d --name qwen38-uncensored-nvfp4 --gpus all --ipc=host --network host \
  --shm-size=16g -v /path/to/models:/models \
  vllm/vllm-openai:v0.27.1 \
  --model /models/Qwen3.8-27B-Uncensored-NVFP4 \
  --served-model-name qwen3.8-27b-uncensored-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 98304 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.945 --enable-prefix-caching --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`--max-model-len 98304` is deliberate: the MTP draft chain reserves KV, and available KV
shrinks with depth (4.11 / 3.83 / 3.79 GiB at k=0/2/3 on a 32 GB card). If a depth refuses
to start, read the `estimated maximum model length` out of the `ValueError` and retry just
under it.

**Concurrency ladder** — vLLM v0.27.1, ctx 98304, 512 output tokens per request,
tok/s **per stream**:

| streams | no MTP | **MTP k=3** | gain |
|---|---|---|---|
| 1 | 61.34 | **80.71** | **+31.6%** |
| 2 | 55.71 | **77.61** | **+39.3%** |
| 4 | 54.86 | **71.80** | **+30.9%** |
| 8 | 56.31 | **71.50** | +27.0% |
| 16 | 52.98 | 68.64 | +29.6% |
| 32 | 52.92 | 68.30 | +29.1% |
| 64 | 52.77 | 66.38 | +25.8% |

**Aggregate** tok/s — where the trade-off lives:

| streams | no MTP | MTP k=3 |
|---|---|---|
| 1 | 60.96 | **79.56** |
| 2 | 109.68 | **145.00** |
| 4 | 213.65 | **230.28** |
| 8 | **434.67** | 307.52 |
| 16 | **811.80** | 378.51 |
| 64 | **826.73** | 430.25 |

Draft acceptance 34.1%.

**Use MTP k=3 at ≤4 concurrent streams; disable it beyond 8** if you need aggregate
throughput. One stream per agent is the case where it pays most.

**Prefill does not collapse** under MTP on this checkpoint:

| prompt | no MTP | MTP k=3 |
|---|---|---|
| 4,096 | 6,344.6 tok/s | 5,981.8 tok/s |
| 16,384 | 10,089.5 tok/s | 7,707.2 tok/s |
| 65,536 | 6,548.4 tok/s | 6,093.6 tok/s |

Worth stating because vLLM logs `max_num_scheduled_tokens is set to 2048 based on the
speculative decoding settings` — enabling MTP silently cuts the prefill chunk 8192 → 2048.
That is the mechanism behind reports of prefill collapsing under speculation. Raising
`--max-num-batched-tokens` does **not** help here: 16384 was slower and 32768 failed to
start with `No available memory for the cache blocks`.

### DGX Spark GB10 (sm_121, 121 GB unified)

```bash
docker run -d --name qwen38-uncensored-nvfp4 --gpus all --network host --ipc host \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/model:/model:ro -v /path/to/cache:/cache \
  -e VLLM_CACHE_ROOT=/cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  <your sm_121 vllm image> \
  /model --served-model-name qwen3.8-27b-uncensored-nvfp4 \
  --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 --max-num-seqs 32 --max-model-len 262144 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The full 262,144 context fits with room to spare: **1,304,862 KV tokens** at util 0.60.
The sm_121a arch environment variables matter — without them FlashInfer and the CUTLASS
DSL kernels will not build for this target.

| | |
|---|---|
| single stream, MTP k=3 | **18.01 tok/s** |
| aggregate @ 8 streams, k=3 | 96.13 tok/s |
| draft acceptance | 33.9% |
| KV tokens @ util 0.60 | 1,304,862 |

**MTP depth does not transfer between GPUs.** On the RTX 5090 single-stream throughput
rises monotonically with depth and k=3 is best. On GB10 the same family peaks at **k=1**
and falls *below* the no-MTP baseline by k=5 — GB10 is bandwidth-bound (~273 GB/s) and
lacks the compute for deep draft chains whose tokens mostly get rejected. Acceptance falls
with depth on both. **Sweep `num_speculative_tokens` on your own hardware.**

vLLM **0.28 does not run under WSL2** — every 0.28 build fails with
`RuntimeError: UVA is not available`. Use 0.27.x there.

## Best practices

Inherited from the base model unless noted.

- **Sampling.** Thinking mode: `temperature=1.0, top_p=0.95, top_k=20, min_p=0`.
  Non-thinking: `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`.
- **Thinking is on by default**, and `reasoning_effort` accepts only `xhigh` (default),
  `medium`, `low` — there is no `"max"`; an unrecognised value raises. Toggle per request:
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- **Tool calling** works on both template branches (thinking on and off) with
  `--tool-call-parser qwen3_xml`. Verify on your own stack before a long agent run: a model
  that cannot emit a parseable call does not fail fast, it burns its whole step budget.
- **Beyond 262,144 tokens**, apply YaRN as the base model card describes. Static YaRN
  degrades short-context quality, so enable it only when you need the range.

## Limitations

Inherited from the abliteration:

- **Refusals are reduced, not eliminated.** 12/100 on the upstream author's heldout set,
  chosen from a 23-point Pareto front over 200 Heretic trials at KL 0.1191. A meaningful
  fraction of harmful requests are still refused.
- **Measured in non-thinking mode, on 100 prompts from one dataset.** Refusal behaviour
  with thinking enabled, and on other topics, is uncharacterized.
- **Behaviour near the old refusal boundary is less stable** than the base model's.
- **Capability:** a 0.5-point mean drop vs base across MMLU, ARC-Challenge, HellaSwag and
  Winogrande, per the upstream card. No generative, math, code or multilingual evaluation
  was run there.

Added by this quantization:

- **+0.0874 PPL** over the bf16 parent on wikitext-2. Quantization compounds whatever
  capability cost the abliteration already carries; neither has been measured on a
  capability benchmark here.
- **No SWE-bench, HumanEval, MMLU or GSM8K run on this checkpoint.**
- **No non-abliterated control.** Until base Qwen3.8-27B is converted with this exact
  recipe and measured on the same harness, nothing here isolates what abliteration costs
  from what quantization costs.
- **Calibration was wikitext only** — generic text, no instruction or task data. It fits
  activation scales; it does not shift behaviour.

## Intended use

Authorized security testing, red-team evaluation, refusal-mechanism and alignment research,
robustness evaluation, and controlled experiments — in isolated environments.

**Out of scope:** deployment to end users or any public endpoint without an independent
safety and moderation layer; generating content intended to harm, harass, defraud or
endanger; any use prohibited by the base model's acceptable use policy.

## Reproducing

Recipe, scripts, benchmarks and the failures along the way:
**https://github.com/tsarihan/qwen3.8-27b-uncensored-nvfp4-rtx5090-gb10-playbook**

## Attribution and license

**Apache-2.0**, inherited from `Qwen/Qwen3.8-27B`. The base model's license and acceptable
use policy still apply to your use of this derivative.

- Abliteration and bf16 weights —
  [JonathanColetti/Qwen3.8-27B-Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored),
  using [Heretic](https://github.com/p-e-w/heretic)
- Base model — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Quantization — [llm-compressor](https://github.com/vllm-project/llm-compressor) /
  [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- Calibration and perplexity corpus — `Salesforce/wikitext`, wikitext-2-raw-v1

Benchmarks were measured on the author's own hardware and are reported as measured,
including the configurations that failed and the hypotheses that turned out to be wrong.
