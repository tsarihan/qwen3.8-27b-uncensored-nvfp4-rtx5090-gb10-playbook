"""Graft the MTP head into v3 and declare it unquantized, matching the reference.

transformers drops the 15 mtp.* tensors again (Qwen3_5ForConditionalGeneration has no MTP
submodule), so they must be re-attached after quantization. Two separate things are
needed and BOTH were missed on the first attempt:

  1. the shard itself, registered in the index, or the weights simply are not there;
  2. an ignore entry, or vLLM builds Qwen3_5MultiTokenPredictor with QUANTIZED slots
     (fc.weight_packed / weight_scale / weight_global_scale) and dies on the BF16
     mtp.fc.weight with "There is no module or parameter named 'fc.weight'".

The reference build declares it as the regex `re:^mtp.*`, which is what is used here
rather than enumerating modules -- it survives any future MTP layer being added.
"""
import json, os, shutil, time

SRC = os.path.expanduser("~/models/Qwen3.8-27B-Uncensored-JonathanColetti")
OUT = os.path.expanduser("~/models/Qwen3.8-27B-Uncensored-JC-NVFP4-v3")
MTP = "model-mtp.safetensors"

from safetensors import safe_open

src_mtp = os.path.join(SRC, MTP)
dst_mtp = os.path.join(OUT, MTP)

with safe_open(src_mtp, framework="pt") as f:
    mtp_keys = list(f.keys())
    dts = {str(f.get_slice(k).get_dtype()) for k in mtp_keys}
print("MTP shard: %d tensors, dtypes=%s" % (len(mtp_keys), dts))
assert all(k.startswith("mtp.") for k in mtp_keys)

if not os.path.exists(dst_mtp):
    shutil.copy2(src_mtp, dst_mtp)
    print("copied %s (%.2f GB)" % (MTP, os.path.getsize(dst_mtp) / 1e9))
else:
    print("%s already present" % MTP)

idx_p = os.path.join(OUT, "model.safetensors.index.json")
idx = json.load(open(idx_p))
wm = idx["weight_map"]
before = len(wm)
for k in mtp_keys:
    wm[k] = MTP
total = sum(os.path.getsize(os.path.join(OUT, f))
            for f in set(wm.values()) if os.path.exists(os.path.join(OUT, f)))
idx["metadata"]["total_size"] = total
shutil.copy2(idx_p, idx_p + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
json.dump(idx, open(idx_p, "w"), indent=1)
print("index: %d -> %d tensors across %d shards, %.2f GB"
      % (before, len(wm), len(set(wm.values())), total / 1e9))

cfg_p = os.path.join(OUT, "config.json")
cfg = json.load(open(cfg_p))
q = cfg["quantization_config"]
ig = list(q.get("ignore", []))
if not any("mtp" in str(x).lower() for x in ig):
    ig.append(r"re:^mtp.*")
    q["ignore"] = ig
    shutil.copy2(cfg_p, cfg_p + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    print("ignore: added 're:^mtp.*' -> %d entries" % len(ig))
else:
    print("ignore already mentions mtp")

print("GRAFT_V3_DONE")
