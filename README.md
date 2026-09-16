# Qwen3.8-27B-Uncensored → NVFP4, on RTX 5090 and DGX Spark (GB10)

Converting `JonathanColetti/Qwen3.8-27B-Uncensored` (bf16) to **NVFP4 ourselves**, and
serving it with **MTP speculative decoding** on two very different Blackwell parts: a
32 GB RTX 5090 (SM120) and a 121 GB DGX Spark GB10 (sm_121).

The reason this repository exists is a result we did not expect:

> **A weight-only NVFP4 conversion was 4.8× worse in perplexity than a full W4A4
> conversion of the same weights — and needle-in-a-haystack could not tell them apart.**

The variable was never W4A4 vs weight-only. It was **which layers get 4 bits**. Copying
the reference checkpoint's `quantization_config` verbatim produced a model that is
simultaneously **more accurate and faster** than the "safer" weight-only build.

---

## Scope and intended use

This is an **abliterated** (refusal-removed) checkpoint. Per the upstream author's own
measurements, refusal drops from 98/100 to 12/100 prompts, and the card states refusal
behaviour is "substantially reduced, not eliminated."

**This repository documents quantization, serving and measurement only.** It is published
to support authorized security testing, red-team evaluation and refusal-mechanism
research in isolated environments. It contains no harmful content, no jailbreak material
and no capability demonstrations — only a quantization recipe, engine configuration and
benchmark numbers.

Do not deploy an abliterated model to end users or any public endpoint without an
independent safety and moderation layer. `Qwen/Qwen3.8-27B` is the correct choice for
essentially every other purpose, and this recipe applies to it unchanged.

## Headline

**Accuracy** — wikitext-2, 24 paired chunks of 2048 tokens, measured through the live
vLLM serving path so the actual FP4/FP8 kernels are in the loop:

| model | PPL | vs bf16 parent |
|---|---|---|
| bf16 parent | 6.8387 | baseline |
| NVFP4**A16** (weight-only, all 496 linears) | 7.2574 | **+0.4187 (+6.12%)** |
| **NVFP4 W4A4 (reference targeting)** | **6.9261** | **+0.0874 (+1.28%)** |

**Speed** — RTX 5090, vLLM v0.27.1, ctx 98304, MTP k=3 vs no speculation:

| conc | no MTP tok/s | **k=3 tok/s** | per-stream gain |
|---|---|---|---|
| 1 | 61.34 | **80.71** | **+31.6%** |
| 2 | 55.71 | **77.61** | **+39.3%** |
| 4 | 54.86 | **71.80** | **+30.9%** |
| 8 | 56.31 | 71.50 | +27.0% |
| 64 | 52.77 | 66.38 | +25.8% |

MTP k=3 is faster **per stream at every concurrency**, and wins on **aggregate** too up to
4 streams (79.6 / 145.0 / 230.3 vs 61.0 / 109.7 / 213.7). Past 8 streams aggregate
inverts — see "When to use MTP".

## The finding: targeting, not weight-only

Our first build used the obvious-looking safe option: `NVFP4A16`, weight-only, activations
left at BF16, `targets="Linear"`. It quantized all 496 language-model linears to 4 bits.

The reference build (`Qwen3.8-27B-NVFP4`) does something quite different:

| component | reference | our first attempt |
|---|---|---|
| MLP gate/up/down, layers 0-55 | **NVFP4** (W4A4, group 16) | NVFP4 |
| **MLP layers 56-63** | **FP8** — sensitivity carve-out | NVFP4 |
| self_attn q/k/v/o | **FP8** W8A8, per-channel | NVFP4 |
| `linear_attn` in_proj_qkv / in_proj_z / out_proj | **FP8** | NVFP4 |
| `linear_attn` **in_proj_a / in_proj_b** | **not quantized at all** | NVFP4 |
| lm_head | **FP8** | BF16 |
| MTP head | BF16, `re:^mtp.*` ignored | BF16 |
| vision tower | BF16, ignored | BF16 |
| KV cache | **static calibrated FP8** | undeclared |

Only **layers 0-55 MLP** actually get 4 bits. Pushing attention and the Gated DeltaNet
projections to FP4 is what cost 6% perplexity. FP4 **activations** cost almost nothing by
comparison — the intuition that weight-only must be gentler is simply wrong here.

Two more things fell out of matching the reference:

* **The FP4 tensor cores came back.** A native FP4 GEMM needs *both* operands in FP4, so
  the A16 build fell back to `MarlinNvFp4LinearKernel`. W4A4 selects
  `FlashInferCutlassNvFp4LinearKernel` — on **both** the 5090 and GB10, so the fallback was
  the scheme, never sm_121.
* **A vLLM accuracy warning disappeared.** The A16 build triggered
  *"the weight global scale is different for parallel layers ... will likely result in
  reduced accuracy"*. It was pointing at the Gated DeltaNet `in_proj_*` projections, whose
  natural scales span 7× (6,176 → 45,056). The reference doesn't quantize them; once we
  stopped, the warning went with it.

**So: read the reference checkpoint's `quantization_config` and copy its `config_groups`,
`targets` and `ignore` verbatim.** `scripts/inspect_base.py` dumps exactly that.

Verified match at tensor level, not just the config header:

| | reference | ours |
|---|---|---|
| packed FP4 tensors | 168 | **168** |
| FP8 scales | 400 | **400** |
| global scales | 168 | **168** |
| plain BF16 weights | 585 | **585** |
| vision | 333 BF16 | **333 BF16** |
| lm_head | F8_E4M3 | **F8_E4M3** |
| ignore entries | 303 | **303** |
| total | ~23.4 GB | **23.42 GB** |

## NIAH cannot detect this

The 6%-worse A16 build scored **20/20 needles at every depth out to 248,419 tokens**.
Needle retrieval is insensitive to this damage class. If you validate a quantization on
NIAH alone you will ship a materially degraded checkpoint and call it verified.

**Measure perplexity against the bf16 parent**, paired over identical chunks.
`scripts/ppl_vllm.py` does it through vLLM's `prompt_logprobs`, so you measure the code
path you actually serve on rather than a transformers-side reimplementation.

For scale: the abliteration itself costs +0.0434 PPL over base Qwen3.8-27B (the upstream
author's measurement). Our 4-bit conversion costs +0.0874 — about double the weight edit,
which is a sane price. The weight-only build's +0.4187 was ~10× the weight edit, and is not.

## The recipe

No retraining, no GPTQ, no AWQ. Weights are round-to-nearest into the FP4/FP8 grids.
Calibration is forward-passes-only and never alters a weight — it exists solely to observe
activation ranges for two things that cannot be derived from weights: the **FP4 activation
global scales** and the **static FP8 KV-cache scales**. 512 samples at seq len 2048.

```python
QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(  # FP8_DYNAMIC
            targets=[
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                r"re:.*lm_head",
                r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
            ], **PRESET_SCHEMES["FP8_DYNAMIC"]),
        "group_1": QuantizationScheme(  # NVFP4
            targets=[r"re:.*mlp\.(gate|up|down)_proj$"],
            **PRESET_SCHEMES["NVFP4"]),
    },
    ignore=[r"re:^mtp.*", r"re:.*visual.*",
            r"re:.*linear_attn\.in_proj_a$", r"re:.*linear_attn\.in_proj_b$",
            r"re:.*linear_attn\.norm$", r"re:.*linear_attn$"],
    kv_cache_scheme=QuantizationArgs(num_bits=8, type="float", strategy="tensor",
                                     symmetric=True, dynamic=False,
                                     observer="static_minmax"),
)
```

Full script: `scripts/quantize_v3.py`. On one GB10 this takes **57 minutes**.

### The MTP head must be grafted back, twice over

`Qwen3_5ForConditionalGeneration` has **no MTP submodule**, so `from_pretrained` discards
all 15 `mtp.*` tensors as unexpected keys and `save_pretrained` never writes them. The
`ignore` pattern is irrelevant — you cannot ignore what was never loaded. Verify with a
tensor count, not an assumption; ours went 15 → 0 silently.

Grafting the shard back is necessary but **not sufficient**. vLLM reads
`quantization_config` to decide which parameter slots to build, so without an ignore entry
it constructs `Qwen3_5MultiTokenPredictor` with *quantized* slots and dies on the BF16
weights:

```
ValueError: There is no module or parameter named 'fc.weight' in
Qwen3_5MultiTokenPredictor. The available parameters belonging to fc
(ColumnParallelLinear) are: {'fc.weight_global_scale', 'fc.weight_scale',
'fc.weight_packed'}
```

Both steps are in `scripts/graft_v3.py`. The reference declares it as `re:^mtp.*`.

MTP stays **BF16** — that is what the reference does, and what orcarouter does. Quantizing
a draft head lowers acceptance, which costs the exact speedup MTP exists to provide, to
save ~0.6 GB on a 23 GB model.

## When to use MTP

**RTX 5090** (ctx 98304, `--max-num-batched-tokens 8192`), aggregate tok/s:

| conc | no MTP | k=3 |
|---|---|---|
| 1 | 60.96 | **79.56** |
| 2 | 109.68 | **145.00** |
| 4 | 213.65 | **230.28** |
| 8 | **434.67** | 307.52 |
| 16 | **811.80** | 378.51 |
| 64 | **826.73** | 430.25 |

Acceptance 34.1%. **Use k=3 at ≤4 streams; disable MTP past 8** if you need aggregate
throughput. If you run one stream per agent, k=3 is a straight +31.6%.

**Prefill does not collapse** on this checkpoint — 5,982 / 7,707 / 6,094 tok/s at
4K/16K/64K with k=3, against 6,345 / 10,090 / 6,548 without. Worth stating because vLLM
logs `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings`,
i.e. MTP silently cuts the prefill chunk 8192 → 2048. That is the real mechanism behind
reports of "chunked prefill collapse" under MTP. It did not bite here.

**Raising `--max-num-batched-tokens` does not help** — we tested it because the chunk
shrink suggested it should:

| batched | result |
|---|---|
| 8192 (default) | **best** — 80.71 tok/s @ c=1 |
| 16384 | worse: 77.31 @ c=1, prefill at 16K down to 2,368, 64K probe 400s |
| 32768 | **fails to start** — `No available memory for the cache blocks` |

**GB10 depth preference is different from the 5090's** and does not transfer. On the
weight-only build, single-stream peaked at **k=1** (16.45 tok/s) and fell below the no-MTP
baseline by k=5, while the 5090 rises monotonically to k=3. GB10 is bandwidth-bound
(~273 GB/s) and lacks compute for deep draft chains whose tokens mostly get rejected;
acceptance falls 57.9% → 24.8% from k=1 to k=5. **Measure depth on your own hardware.**

## Environment

| | RTX 5090 | DGX Spark GB10 |
|---|---|---|
| GPU | GeForce RTX 5090, 32,607 MiB, SM120 | GB10, 121 GiB unified, sm_121 |
| OS | Ubuntu 24.04 on WSL2 | DGX OS |
| vLLM | `vllm/vllm-openai:v0.27.1` | `vllm-dsv4:src-sm121` (0.25.2.dev0) |
| serving ctx | 98,304 | 262,144 |
| KV tokens | ~100,257 @ util 0.945 | 1,304,862 @ util 0.60 |

vLLM **0.28 does not run on WSL2** — every 0.28 build dies with
`RuntimeError: UVA is not available`, because WSL detection reports pinned memory
unavailable and the 0.28 GPU worker hard-requires a `UvaBuffer`. Use 0.27.x there.

On v0.27.1 vs v0.26.0: the newer release costs ~0.5 GiB of KV on a 32 GB card
(4.62 → 4.11 GiB, max ctx 131,072 → ~123,872) and runs ~3-4% slower on the no-MTP ladder.
It is still the right choice here — it is what the checkpoint family's cards require.

## Reproducing

```bash
# 1. verify the bf16 parent against the publisher's LFS sha256 before converting 55 GB
HF_REPO=JonathanColetti/Qwen3.8-27B-Uncensored \
LOCAL_DIR=~/models/Qwen3.8-27B-Uncensored-JonathanColetti \
  python3 scripts/verify-nvfp4.py            # -> VERIFY_OK 14/14

# 2. read the REFERENCE checkpoint's recipe rather than inventing one
python3 scripts/inspect_base.py "/path/to/Qwen3.8-27B-NVFP4"

# 3. convert (57 min on one GB10)
python3 scripts/quantize_v3.py

# 4. graft the MTP head back AND declare it unquantized
python3 scripts/graft_v3.py

# 5. confirm the fused groups share a global scale (q/k/v, gate/up)
python3 scripts/check-global-scales.py

# 6. accuracy, against the bf16 parent, through the serving path
bash scripts/ppl-compare.sh

# 7. serve
./scripts/serve-gb10.sh                       # GB10, env-var driven
bash scripts/bench-5090.sh                    # 5090: nomtp / k=3 / batched-token sweep
```

## Notes that cost time

* **transformers 5.14.1 cannot save an offloaded, sharded model.**
  `weight_map.update({k: v} for k in ...)` passes a generator of single-entry dicts to
  `dict.update()`, which wants a mapping or key/value pairs:
  `ValueError: dictionary update sequence element #0 has length 1; 2 is required`.
  It only fires when the save shards, so a single-shard run never sees it. One-line fix in
  `patches/fix_transformers_weightmap.py`. We lost a full calibration pass to this.
* **`llmcompressor` infers `DataFreePipeline` for a weight-only scheme regardless of
  whether you pass a dataset**, and that pipeline skips `start_calibration()`, which is the
  only place `fuse_weight_observers()` runs. Forcing `pipeline="basic"` changes it. This
  turned out to be a red herring for us — the Q/K/V and gate/up scales were already shared
  in both builds — but the mechanism is real if you hit it.
* **`pgrep -f "foo.py"` matches the shell running it.** A watcher loop built that way never
  exits, and the same pattern in a `kill` sends the signal to your own shell. Use the
  bracket trick (`pgrep -f "foo[.]py"`) or capture the PID at launch.
* **Deeper MTP needs slightly less context.** Available KV shrinks with draft depth
  (4.11 / 3.83 / 3.79 GiB at k=0/2/3 on a 32 GB card). k=2 failed at ctx 98,304 by
  **0.01 GiB** while the engine reported its own maximum as 97,600 — always read the
  `estimated maximum model length` out of the `ValueError` and retry just under it.
* **Do not pass `--quantization` or `--kv-cache-dtype`.** This checkpoint's `config.json`
  carries the mixed-precision `config_groups` *and* a calibrated static FP8
  `kv_cache_scheme`; both flags override them.

## Files

```
scripts/quantize_v3.py            the recipe: two config groups, matching the reference
scripts/graft_v3.py               re-attach the MTP head and declare it unquantized
scripts/inspect_base.py           dump any checkpoint's per-group quantization reality
scripts/check-global-scales.py    which sibling groups share a global scale
scripts/ppl_vllm.py               perplexity via vLLM prompt_logprobs (the serving path)
scripts/serve-gb10.sh             GB10 launcher, all knobs are env vars
scripts/bench-5090.sh             5090: no-MTP / k=3 / batched-token sweep + prefill probe
scripts/longprompt_probe.py       TTFT and prefill across the chunk boundary
scripts/sweep.py                  concurrency ladder
scripts/needles.py                multi-needle retrieval
scripts/vision_test.py            vision tower survived quantization
scripts/vision_digits.py          OCR by digit length, to separate noise from breakage
scripts/verify-nvfp4.py           checkpoint vs publisher LFS sha256
patches/fix_transformers_weightmap.py   the offloaded+sharded save bug
results/                          raw ladders, prefill probes, PPL comparison
docs/JOURNEY.md                   what failed, in order, and why
```

## License

Apache-2.0, matching the upstream `JonathanColetti/Qwen3.8-27B-Uncensored` and
`Qwen/Qwen3.8-27B`. See `NOTICE` for attribution.
