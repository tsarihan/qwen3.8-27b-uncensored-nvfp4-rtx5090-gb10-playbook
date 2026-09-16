"""v3: NVFP4 W4A4 mixed-precision, matching the reference Qwen3.8-27B-NVFP4 exactly.

v1/v2 were NVFP4A16 -- weight-only. That was the wrong call: with BF16 activations a
native FP4 GEMM is impossible (it needs BOTH operands in FP4), so vLLM fell back to
MarlinNvFp4LinearKernel and the FP4 tensor cores sat idle. It also blanket-quantized
every Linear, including the Gated DeltaNet in_proj_* projections whose natural scales
span 7x -- which is what produced vLLM's "global scale is different for parallel layers"
warning.

This recipe is lifted from the reference build's own quantization_config, which is the
configuration proven to run MTP k=1..7 on this hardware:

  group_0  FP8_DYNAMIC (W8A8, per-channel weights / per-token acts)
           - self_attn q/k/v/o
           - linear_attn in_proj_qkv, in_proj_z, out_proj
           - lm_head
           - layers 56-63 MLP  <- deliberate sensitivity carve-out: the last 8 MLP
             layers get FP8, not FP4
  group_1  NVFP4 (W4A4, group_size 16)
           - all other mlp gate/up/down

  ignored (unquantized BF16), matching the reference's 303-entry list:
           - re:^mtp.*                 the draft head, verbatim
           - vision tower
           - linear_attn in_proj_a, in_proj_b, and the module/norm itself
             ^ this is the fix for the fused-global-scale warning: the reference does
               not quantize these at all.

  kv_cache_scheme  static per-tensor FP8, calibrated.

Calibration is genuinely required here, unlike v1/v2: FP4 activation global scales and
the STATIC FP8 KV scales cannot be computed from weights. 512 samples at seq len 2048,
matching the sample count and length orcarouter used for this model family.
"""
import os, time, json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

SRC = os.path.expanduser("~/models/Qwen3.8-27B-Uncensored-JonathanColetti")
OUT = os.path.expanduser("~/models/Qwen3.8-27B-Uncensored-JC-NVFP4-v3")
NSAMPLES, SEQLEN = 512, 2048

FP8_TARGETS = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
]
NVFP4_TARGETS = [r"re:.*mlp\.(gate|up|down)_proj$"]
IGNORE = [
    r"re:^mtp.*",
    r"re:.*visual.*",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
    r"re:.*linear_attn\.norm$",
    r"re:.*linear_attn$",
]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

print("building %d calibration samples @ %d ..." % (NSAMPLES, SEQLEN), flush=True)
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
texts, buf = [], ""
for t in raw["text"]:
    buf += t
    if len(buf) > SEQLEN * 5:
        texts.append(buf); buf = ""
    if len(texts) >= NSAMPLES:
        break
rows = []
for t in texts[:NSAMPLES]:
    e = tok(t, max_length=SEQLEN, truncation=True)
    if len(e["input_ids"]) >= 256:
        rows.append({"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]})
ds = Dataset.from_list(rows)
print("  %d samples" % len(ds), flush=True)

print("loading bf16 ...", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    SRC, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(targets=FP8_TARGETS, **PRESET_SCHEMES["FP8_DYNAMIC"]),
        "group_1": QuantizationScheme(targets=NVFP4_TARGETS, **PRESET_SCHEMES["NVFP4"]),
    },
    ignore=IGNORE,
    kv_cache_scheme=QuantizationArgs(
        num_bits=8, type="float", strategy="tensor",
        symmetric=True, dynamic=False, observer="static_minmax"),
)

print("quantizing (W4A4 + FP8, calibrated) ...", flush=True)
t1 = time.time()
oneshot(model=model, recipe=recipe, output_dir=OUT,
        dataset=ds, num_calibration_samples=len(ds), max_seq_length=SEQLEN,
        trust_remote_code_model=True)
print("  quantized+saved in %.1f min" % ((time.time() - t1) / 60), flush=True)

try:
    AutoProcessor.from_pretrained(SRC, trust_remote_code=True).save_pretrained(OUT)
    print("  processor saved", flush=True)
except Exception as e:
    print("  processor save skipped:", type(e).__name__, e, flush=True)

cfg = json.load(open(os.path.join(OUT, "config.json")))
q = cfg.get("quantization_config", {})
print("\n  format:", q.get("format"))
print("  groups:", list((q.get("config_groups") or {}).keys()))
print("  kv_cache_scheme set:", q.get("kv_cache_scheme") is not None)
print("  ignore entries:", len(q.get("ignore", [])))
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
print("  output: %.2f GB" % (tot / 1e9))
print("TOTAL %.1f min" % ((time.time() - t0) / 60))
print("QUANTIZE_V3_COMPLETE")
