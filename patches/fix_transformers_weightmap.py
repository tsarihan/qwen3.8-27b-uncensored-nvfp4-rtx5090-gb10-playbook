#!/usr/bin/env python3
"""Fix a transformers 5.14.1 bug that aborts save_pretrained for offloaded+sharded models.

Symptom, after a full calibration pass has already completed:

    File "transformers/modeling_utils.py", line ~3675, in save_pretrained
      weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())
    ValueError: dictionary update sequence element #0 has length 1; 2 is required

`dict.update()` is handed a GENERATOR OF SINGLE-ENTRY DICTS. It accepts a mapping, or an
iterable of (key, value) pairs -- not an iterable of dicts. The closing brace is simply in
the wrong place; the intent is clearly a dict comprehension.

The branch only runs when the model `is_offloaded` AND the save shards, which is why a
single-shard save never trips it. If you quantize a large model with device_map="auto" and
it writes more than one shard, you will hit this.

Usage:
    python3 fix_transformers_weightmap.py [/path/to/site-packages/transformers/modeling_utils.py]

Defaults to the transformers importable in the current interpreter. Writes a timestamped
backup next to the file before changing it.
"""
import shutil
import sys
import time
from pathlib import Path

OLD = "weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())"
NEW = "weight_map.update({k: os.path.basename(shard_file) for k in shard_state_dict.keys()})"


def target_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    import transformers.modeling_utils as mu
    return Path(mu.__file__)


def main() -> int:
    path = target_path()
    if not path.exists():
        print("not found: %s" % path)
        return 1

    src = path.read_text()
    if NEW in src and OLD not in src:
        print("already patched: %s" % path)
        return 0

    n = src.count(OLD)
    print("target      : %s" % path)
    print("occurrences : %d" % n)
    if n != 1:
        print("expected exactly one occurrence; not touching the file")
        return 1

    backup = "%s.bak-weightmap-%s" % (path, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(path, backup)
    path.write_text(src.replace(OLD, NEW, 1))
    print("backup      : %s" % backup)
    print("patched")

    import ast
    ast.parse(path.read_text())
    print("syntax check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
