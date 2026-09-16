#!/usr/bin/env bash
# Perplexity: bf16 parent vs our two quantizations, measured through the live vLLM path.
#
# Three rows, because two of them answer different questions:
#   bf16     the parent -- the only honest baseline for "what did quantization cost"
#   A16-v2   weight-only NVFP4 (BF16 activations)
#   v3       W4A4 mixed-precision matching the reference build
# bf16-vs-v3 is the accuracy cost of the conversion; A16-vs-v3 says whether matching the
# reference (FP4 activations, FP8 attention, the layer 56-63 carve-out) cost or saved
# accuracy relative to the softer weight-only build.
#
# Identical chunks for all three (the harness caches them on first run), so these are
# PAIRED measurements over the same tokens rather than independent samples.
# MTP is OFF everywhere: speculative decoding verifies every token against the target, so
# it cannot change logprobs, but leaving it out removes a variable.
# max-model-len 8192 keeps the 55 GB bf16 comfortable; PPL chunks are 2048 tokens.
set -u
R=$HOME/llmruntimes/qwen3.8-27b-uncensored-jc/results
mkdir -p "$R"
SUM="$R/ppl-compare.log"
: > "$SUM"
PORT=8899
TAG=$(date +%H%M%S)

for c in $(docker ps --format "{{.Names}}" | grep -E "^jc-|^ppl-"); do
  docker stop "$c" >/dev/null 2>&1 && echo "stopped $c" | tee -a "$SUM"
done
sleep 8

run_ppl () {
  local label="$1" dir="$2" util="$3"
  local name="ppl-${label}-${TAG}"
  echo "" | tee -a "$SUM"
  echo "=========== $label  $(date +%T) ===========" | tee -a "$SUM"

  docker run -d --name "$name" --gpus all --network host --ipc host --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$dir":/model:ro \
    -v /data/models/vllm-cache-jc-nvfp4a16:/cache \
    -e VLLM_CACHE_ROOT=/cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang \
    vllm-dsv4:src-sm121 \
    /model --served-model-name pplmodel --host 0.0.0.0 --port $PORT \
    --trust-remote-code --tensor-parallel-size 1 \
    --gpu-memory-utilization "$util" --max-num-seqs 4 --max-model-len 8192 \
    --enable-prefix-caching --enable-chunked-prefill \
    >> "$R/ppl-boot-${label}.log" 2>&1

  local up=0 i
  for i in $(seq 1 140); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    docker ps --filter "name=^${name}$" --format "{{.Status}}" | grep -q Up || break
    sleep 10
  done
  if [ "$up" != "1" ]; then
    echo "  FAILED TO BOOT" | tee -a "$SUM"
    docker logs "$name" 2>&1 | grep -iE "ValueError|RuntimeError|memory" | tail -3 | tee -a "$SUM"
    docker stop "$name" >/dev/null 2>&1; sleep 8; return
  fi
  echo "  booted in ~$((i*10))s" | tee -a "$SUM"
  docker logs "$name" 2>&1 | grep -iE "NVFP4 GEMM" | tail -1 | sed "s/^/  /" | tee -a "$SUM"

  MODEL=pplmodel VLLM_URL="http://127.0.0.1:$PORT" NCHUNK=24 CHUNKTOK=2048 \
    python3 /tmp/ppl_vllm.py "$label" 2>&1 | tail -8 | tee -a "$SUM"

  docker stop "$name" >/dev/null 2>&1 && echo "  stopped (kept)" | tee -a "$SUM"
  sleep 10
}

run_ppl bf16   "$HOME/models/Qwen3.8-27B-Uncensored-JonathanColetti"   0.75
run_ppl A16-v2 "$HOME/models/Qwen3.8-27B-Uncensored-JC-NVFP4A16-v2"    0.60
run_ppl v3-W4A4 "$HOME/models/Qwen3.8-27B-Uncensored-JC-NVFP4-v3"      0.60

echo "" | tee -a "$SUM"
echo "=== SUMMARY ===" | tee -a "$SUM"
grep "PPL_RESULT" "$SUM" | tee -a "$SUM"
python3 - <<'PY' | tee -a "$SUM"
import re, os
p = os.path.expanduser("~/llmruntimes/qwen3.8-27b-uncensored-jc/results/ppl-compare.log")
rows = {}
for line in open(p):
    m = re.match(r"PPL_RESULT (\S+) ([0-9.]+) (\d+)", line.strip())
    if m:
        rows[m.group(1)] = (float(m.group(2)), int(m.group(3)))
base = rows.get("bf16", (None, None))[0]
print("\n%-12s %10s %12s %10s" % ("model", "PPL", "vs bf16", "tokens"))
for k in ("bf16", "A16-v2", "v3-W4A4"):
    if k not in rows: continue
    ppl, n = rows[k]
    d = "" if (base is None or k == "bf16") else "%+.4f (%+.2f%%)" % (ppl - base, 100 * (ppl - base) / base)
    print("%-12s %10.4f %12s %10d" % (k, ppl, d or "baseline", n))
PY
echo "PPL_COMPARE_DONE"
