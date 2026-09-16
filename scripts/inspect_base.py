"""Exactly how does the reference Qwen3.8-27B-NVFP4 quantize each part?

This is the build the GB10 playbook ran MTP k=1..7 against, so whatever it does is
known-good on this hardware. Report per-group: language-model linears, MTP, vision,
lm_head, embeddings -- plus the kv_cache_scheme and the ignore list.
"""
import json, os, sys, glob, collections
from safetensors import safe_open

D = sys.argv[1]
cfg = json.load(open(os.path.join(D, "config.json")))
q = cfg.get("quantization_config", {}) or {}

print("=== quantization_config ===")
print("  quant_method :", q.get("quant_method"))
print("  format       :", q.get("format"))
print("  kv_cache_scheme:", json.dumps(q.get("kv_cache_scheme")))
cg = q.get("config_groups") or {}
for name, g in cg.items():
    w = (g or {}).get("weights") or {}
    ia = (g or {}).get("input_activations")
    tg = (g or {}).get("targets")
    print("  group %s: weights num_bits=%s type=%s strategy=%s group_size=%s | input_act=%s"
          % (name, w.get("num_bits"), w.get("type"), w.get("strategy"),
             w.get("group_size"),
             "None" if ia is None else "num_bits=%s type=%s" % (ia.get("num_bits"), ia.get("type"))))
    print("        targets: %s" % (tg if not isinstance(tg, list) else tg[:6]))
ig = q.get("ignore", [])
print("  ignore entries:", len(ig))
regexy = [x for x in ig if str(x).startswith("re:")]
print("  regex ignores:", regexy[:10])
for tag in ("mtp", "visual", "lm_head", "embed"):
    hits = [x for x in ig if tag in str(x).lower()]
    print("    %-8s in ignore: %s%s" % (tag, len(hits), (" e.g. " + str(hits[:2])) if hits else ""))

idx_p = os.path.join(D, "model.safetensors.index.json")
if os.path.exists(idx_p):
    keys = list(json.load(open(idx_p))["weight_map"])
    shards = {}
    for k, v in json.load(open(idx_p))["weight_map"].items():
        shards.setdefault(v, []).append(k)
else:
    keys, shards = [], {}
    for p in glob.glob(os.path.join(D, "*.safetensors")):
        with safe_open(p, framework="pt") as f:
            ks = list(f.keys())
        keys += ks
        shards[os.path.basename(p)] = ks

def group_of(k):
    kl = k.lower()
    if kl.startswith("mtp") or ".mtp." in kl or "nextn" in kl: return "MTP"
    if "visual" in kl: return "vision"
    if "lm_head" in kl: return "lm_head"
    if "embed_tokens" in kl: return "embeddings"
    return "language_model"

print("\n=== per-group tensor kinds ===")
kinds = collections.defaultdict(collections.Counter)
dtypes = collections.defaultdict(collections.Counter)
openf = {}
for shard, ks in shards.items():
    p = os.path.join(D, shard)
    f = safe_open(p, framework="pt")
    for k in ks:
        g = group_of(k)
        if k.endswith("weight_packed"):   kinds[g]["packed(FP4)"] += 1
        elif k.endswith("weight_scale"):  kinds[g]["scale"] += 1
        elif k.endswith("weight_global_scale"): kinds[g]["global_scale"] += 1
        elif k.endswith("weight_scale_inv") or k.endswith("input_scale"): kinds[g]["other_scale"] += 1
        elif k.endswith(".weight"):       kinds[g]["plain .weight"] += 1
        else:                             kinds[g]["misc"] += 1
        try:
            dtypes[g][str(f.get_slice(k).get_dtype())] += 1
        except Exception:
            pass

for g in ("language_model", "MTP", "vision", "lm_head", "embeddings"):
    if g not in kinds: continue
    print("  %-14s %s" % (g, dict(kinds[g])))
    print("  %-14s dtypes: %s" % ("", dict(dtypes[g])))
