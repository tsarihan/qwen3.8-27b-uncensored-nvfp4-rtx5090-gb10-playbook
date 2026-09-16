"""Find which sibling group actually has mismatched global scales, if any."""
import os, re, collections
from safetensors import safe_open

P = os.path.expanduser("~/models/Qwen3.8-27B-Uncensored-JC-NVFP4A16/model.safetensors")

with safe_open(P, framework="pt") as f:
    gs = {k: float(f.get_tensor(k).reshape(-1)[0])
          for k in f.keys() if k.endswith("weight_global_scale")}

# group by (layer, parent module) -> {leaf: scale}
groups = collections.defaultdict(dict)
for k, v in gs.items():
    m = re.match(r"(.*\.layers\.\d+\.[a-z_]+)\.([a-z0-9_]+)\.weight_global_scale$", k)
    if m:
        groups[m.group(1)][m.group(2)] = v

mismatched = collections.Counter()
matched = collections.Counter()
examples = {}
for parent, leaves in groups.items():
    if len(leaves) < 2:
        continue
    fam = re.sub(r"\.layers\.\d+\.", ".layers.N.", parent)
    vals = list(leaves.values())
    if all(abs(x - vals[0]) < 1e-9 for x in vals):
        matched[fam] += 1
    else:
        mismatched[fam] += 1
        examples.setdefault(fam, (parent, dict(leaves)))

print("=== sibling groups with SHARED global scale ===")
for fam, c in matched.most_common():
    print("  %-55s %d groups" % (fam, c))
print("\n=== sibling groups with DIFFERENT global scale ===")
if not mismatched:
    print("  none")
for fam, c in mismatched.most_common():
    print("  %-55s %d groups" % (fam, c))
    parent, leaves = examples[fam]
    print("     e.g. %s" % parent)
    for leaf, v in leaves.items():
        print("        %-16s %.8g" % (leaf, v))

allv = set(round(v, 6) for v in gs.values())
print("\n  total global_scale tensors: %d | distinct values: %d" % (len(gs), len(allv)))
print("  sample values:", sorted(allv)[:8])
