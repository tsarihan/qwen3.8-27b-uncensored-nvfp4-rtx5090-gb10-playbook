# Journey: what failed, in order

Kept because the failures are more reusable than the final recipe. The largest one was a
design decision that looked conservative, passed every test we ran, and was wrong.

## 0. Why convert at all

`JonathanColetti/Qwen3.8-27B-Uncensored` publishes bf16 and GGUF, but **no NVFP4**. The
only NVFP4 derived from those weights is a third-party requant by someone other than the
author of the weight edit — an additional, unauditable reprocessing of every tensor.

Converting it ourselves is the shortest provenance chain available: the author's own bf16,
sha256-verified against their LFS hashes (`VERIFY_OK`, 14/14), then one conversion we
control and can describe.

## 1. The weight-only build: conservative-looking, and 6% worse

The first conversion used `NVFP4A16` with `targets="Linear"` — weight-only, activations
left at BF16. The reasoning was that quantizing fewer things must be gentler, so it should
sit closer to bf16.

It scored **20/20 needles out to 248,419 tokens** and passed a vision check. Everything we
measured said it was fine.

Then we measured perplexity against the bf16 parent:

```
  bf16 parent   6.8387
  NVFP4A16      7.2574   +0.4187  (+6.12%)
  W4A4          6.9261   +0.0874  (+1.28%)
```

**Weight-only was 4.8× worse than quantizing more aggressively.** The variable was never
A16-vs-W4A4 — it was *which* layers got 4 bits. `targets="Linear"` put all 496 language
model linears at FP4, including self-attention, the Gated DeltaNet projections, and the
last 8 MLP layers. The reference build protects all of those at FP8 or leaves them
unquantized, and only FP4s layers 0-55 MLP.

Two things followed from fixing it:

* the FP4 tensor cores came back (`FlashInferCutlassNvFp4LinearKernel` instead of
  `MarlinNvFp4LinearKernel`) — a native FP4 GEMM needs *both* operands in FP4, so
  weight-only can never use them, on either the 5090 or GB10;
* a vLLM warning about mismatched global scales in fused parallel layers disappeared,
  because the layers it was complaining about are ones the reference never quantizes.

**Lesson:** read the reference checkpoint's `quantization_config` and copy its
`config_groups`, `targets` and `ignore` verbatim. Do not reason from first principles about
which scheme "should" be gentler, and do not trust NIAH to catch the difference.

## 2. The MTP head vanished, silently

`Qwen3_5ForConditionalGeneration` has no MTP submodule, so `from_pretrained` treated all
15 `mtp.*` tensors as unexpected keys and dropped them; `save_pretrained` then wrote a
checkpoint without them. The `ignore` pattern was irrelevant — you cannot ignore what was
never loaded.

Nothing errored. The only reason we caught it was counting tensors by family in the output
and seeing `mtp: 15 -> 0`.

## 3. Grafting the head back was necessary but not sufficient

With the shard re-attached and registered in the index, every MTP depth still failed:

```
ValueError: There is no module or parameter named 'fc.weight' in
Qwen3_5MultiTokenPredictor. The available parameters belonging to fc
(ColumnParallelLinear) are: {'fc.weight_global_scale', 'fc.weight_scale',
'fc.weight_packed'}
```

vLLM reads `quantization_config` to decide which parameter slots to build. `llmcompressor`
writes the `ignore` list from modules it actually saw — and MTP was never loaded, so
nothing recorded it. vLLM therefore built quantized slots for a BF16 shard. Adding the
modules to `ignore` fixed it; the reference declares them as `re:^mtp.*`.

Both halves are needed: **the weights, and the declaration.**

## 4. Chasing a warning down the wrong path

vLLM warned that "the weight global scale is different for parallel layers (e.g. q_proj,
k_proj, v_proj)". We assumed our checkpoint was defective, worked out that
`fuse_weight_observers()` only runs inside `start_calibration()`, that `llmcompressor`
infers `DataFreePipeline` for a weight-only scheme regardless of whether a dataset is
supplied, and rebuilt the whole checkpoint with `pipeline="basic"` to force it.

Then we inspected the tensors. **Q/K/V were already sharing a global scale in both builds**
(5376/5376/5376), as were gate/up (6400/6400). The rebuild changed nothing.

The warning was real but pointed elsewhere: the Gated DeltaNet `in_proj_a` / `in_proj_b` /
`in_proj_qkv` / `in_proj_z` / `out_proj` projections, whose natural scales span 7×
(6,176 → 45,056). Forcing a shared scale across that range would have made things worse.
The actual fix was to stop quantizing them, which is what the reference does.

**Lesson:** read the artifact before rewriting the toolchain. Two hours would have been
one `safetensors` read.

## 5. transformers 5.14.1 cannot save an offloaded, sharded model

After a full 57-minute calibration pass, the save died:

```
File "transformers/modeling_utils.py", line 3675, in save_pretrained
  weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```

`dict.update()` is handed a generator of single-entry dicts; the closing brace is in the
wrong place and it should be a dict comprehension. The branch only runs when the model is
offloaded **and** the save shards — the earlier weight-only builds wrote a single shard and
never hit it.

One-line fix in `patches/fix_transformers_weightmap.py`. Cost: one full calibration pass.

## 6. Container names, and a retry ladder that was 704 tokens too coarse

Two self-inflicted stalls worth recording because both present as "the config doesn't work":

* Containers are never removed in this workflow, so a fixed name collides with the exited
  container from the previous attempt and `docker run` fails in ~5 seconds. Every depth in
  an MTP ladder needs a unique name.
* MTP's draft chain reserves KV, so available KV shrinks with depth: 4.11 / 3.83 / 3.79 GiB
  at k=0/2/3 on a 32 GB card. Our retry ladder went 131072 → 98304, and **k=2 needed
  3.84 GiB against 3.83 available** — it failed by 0.01 GiB while the engine reported its
  own maximum as 97,600. Read `estimated maximum model length` out of the `ValueError` and
  retry just under it, rather than guessing round numbers.

## 7. A hypothesis that was reasonable and still wrong

Enabling MTP makes vLLM log `max_num_scheduled_tokens is set to 2048 based on the
speculative decoding settings` — speculation silently cuts the prefill chunk 8192 → 2048.
That is a real mechanism and it explains published reports of prefill collapsing under MTP.

We predicted raising `--max-num-batched-tokens` ~4× would restore it and let k=3 win at
concurrency. It did not:

| batched | outcome |
|---|---|
| 8192 (default) | **best**: 80.71 tok/s @ c=1 |
| 16384 | worse: 77.31 @ c=1; prefill at 16K falls to 2,368; the 64K probe 400s |
| 32768 | **fails to start**: `No available memory for the cache blocks` |

On this checkpoint the default was already correct, and prefill never collapsed to begin
with (5,982 / 7,707 / 6,094 tok/s at 4K/16K/64K under k=3). We had reasoned from a
different checkpoint's numbers to a lever this one did not need.

## 8. MTP depth does not transfer between machines

The 5090 and GB10 disagree about the best draft depth, in opposite directions:

* **5090** — single-stream rises monotonically with depth, +31.6% at k=3.
* **GB10** — peaks at **k=1** (16.45 tok/s) and falls *below* the no-MTP baseline by k=5.

Acceptance falls with depth on both (57.9% → 24.8% on GB10, 58.1% → 34.1% on the 5090).
GB10 is bandwidth-bound at ~273 GB/s and lacks the compute to run deep draft chains whose
tokens mostly get rejected; the 5090 has compute to spare and still profits at 34%
acceptance. Measure depth on the hardware you will serve on.

## 9. Known gaps

* **No SWE-bench Pro run on this checkpoint.** The 85.0% figure in our other playbook is a
  *different* abliteration (orcarouter's) on a different quantization.
* **No non-abliterated control.** Until base `Qwen3.8-27B` is converted with this exact
  recipe and measured on the same harness, nothing here says what abliteration costs.
* **PPL uses 24 chunks of 2048 tokens**, not a whole-corpus sweep. Rows are paired and
  therefore comparable to each other, but not directly to numbers computed over a different
  amount of text — including the upstream author's own tables.
* **The calibration corpus is wikitext**, generic text only. No instruction data, no task
  data, nothing that could shift behaviour: it is used solely to observe activation ranges
  for the FP4 activation global scales and the static FP8 KV scales.
