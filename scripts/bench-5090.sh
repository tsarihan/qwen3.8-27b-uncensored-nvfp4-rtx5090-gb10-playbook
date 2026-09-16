#!/usr/bin/env bash
# Goal: make MTP k=3 BEAT MTP-off for Jonathan's model on the RTX 5090.
#
# Four configs, changing one thing at a time:
#   A  no MTP,  batched  8192   -- the bar to beat
#   B  k=3,     batched  8192   -- MTP at the stock chunk size
#   C  k=3,     batched 32768   -- the lever
#   D  k=3,     batched 16384   -- fallback if C will not fit
#
# Why C is the lever: enabling MTP makes vLLM log
#   "max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings"
# i.e. speculation silently cuts the prefill chunk 8192 -> 2048, so prompts cross a chunk
# boundary 4x sooner and every chunk carries draft overhead. That -- not "chunked prefill
# is broken" -- is what produced the >20x prefill collapse in the RTX5090 playbook's
# Finding A. Raising max-num-batched-tokens ~4x should restore the effective chunk.
#
# No --quantization and no --kv-cache-dtype: this checkpoint's config.json carries the
# mixed-precision config AND a calibrated static FP8 kv_cache_scheme, and passing those
# flags would override them.
#
# Containers are stopped, never removed.
set -u
M=/models/Qwen3.8-27B-Uncensored-JC-NVFP4-v3
RT=$HOME/llmruntimes/qwen3.8-27b-uncensored-jc
R=$RT/results-5090-v3
mkdir -p "$R"
SUM="$R/summary.log"
: > "$SUM"
PORT=8899
MODEL=qwen3.8-27b-uncensored-jc
TAG=$(date +%H%M%S)

echo "=== waiting for transfer ===" | tee -a "$SUM"
while pgrep -f "rsync -a --info=progress2" >/dev/null 2>&1; do sleep 20; done
sleep 5
du -sh $HOME/models/Qwen3.8-27B-Uncensored-JC-NVFP4-v3 | tee -a "$SUM"

for c in $(docker ps --format "{{.Names}}" | grep -E "^jc5090|^orca-|vllm-qwen3.8-27b-orca"); do
  docker stop "$c" >/dev/null 2>&1 && echo "stopped $c" | tee -a "$SUM"
done
sleep 8

run_cfg () {
  local label="$1" k="$2" batched="$3"
  local name="jc5090-${label}-${TAG}"
  echo "" | tee -a "$SUM"
  echo "=========== $label : mtp=${k:-off} batched=$batched  $(date +%T) ===========" | tee -a "$SUM"

  local SPEC=()
  [ -n "$k" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${k}}")

  local ctx
  for ctx in 98304 65536; do
    docker run -d --name "$name" --gpus all --ipc=host --network host --shm-size=16g \
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
      -v $HOME/models:/models \
      -v $HOME/router/q38cache/flashinfer:/root/.cache/flashinfer \
      -v $HOME/router/q38cache/vllm:/root/.cache/vllm \
      vllm/vllm-openai:v0.27.1 \
      --model "$M" --host 0.0.0.0 --port $PORT --served-model-name "$MODEL" \
      --max-model-len "$ctx" --max-num-seqs 16 --max-num-batched-tokens "$batched" \
      --gpu-memory-utilization 0.945 --enable-prefix-caching --trust-remote-code \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
      "${SPEC[@]}" >> "$R/boot-${label}.log" 2>&1
    local up=0 i
    for i in $(seq 1 130); do
      curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
      docker ps --filter "name=^${name}$" --format "{{.Status}}" | grep -q Up || break
      sleep 10
    done
    [ "$up" = "1" ] && { echo "booted ctx=$ctx in ~$((i*10))s" | tee -a "$SUM"; break; }
    echo "  failed at ctx=$ctx" | tee -a "$SUM"
    docker logs "$name" 2>&1 | grep -iE "estimated maximum|ValueError" | tail -2 | tee -a "$SUM"
    docker stop "$name" >/dev/null 2>&1; sleep 8
    name="${name}-c65k"
  done
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "  GAVE UP" | tee -a "$SUM"; return; }

  docker logs "$name" 2>&1 | grep -iE "NVFP4 GEMM" | tail -1 | sed "s/^/  /" | tee -a "$SUM"
  docker logs "$name" 2>&1 | grep -iE "max_num_scheduled_tokens is set to" | tail -1 | sed "s/^/  /" | tee -a "$SUM"

  curl -s -m 200 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with just the number.\"}],\"max_tokens\":400,\"temperature\":0}" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('  correctness (391):',((m.get('content') or m.get('reasoning_content') or '')).strip()[:40])" \
    2>&1 | tee -a "$SUM"

  A=$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics" 2>/dev/null)
  python3 "$HOME/sweep.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --concurrency 1,2,4,8,16,32,64 --max-tokens 512 --tag "$label" \
    --out "$R/sweep-${label}.json" > "$R/sweep-${label}.log" 2>&1
  B=$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics" 2>/dev/null)
  tail -11 "$R/sweep-${label}.log" | tee -a "$SUM"

  echo "  --- prefill probe ---" | tee -a "$SUM"
  LABEL="$label" VLLM_URL="http://127.0.0.1:$PORT" MODEL="$MODEL" \
    python3 /tmp/longprompt_probe.py 2>&1 | tee -a "$SUM"

  python3 - <<PY | tee -a "$SUM"
import re
a = """$A"""; b = """$B"""
def g(t,n):
    m = re.search(r'^vllm:%s\{[^}]*\}\s+([0-9.e+-]+)' % n, t, re.M)
    return float(m.group(1)) if m else None
p=[g(a,"spec_decode_num_accepted_tokens_total"),g(a,"spec_decode_num_draft_tokens_total"),
   g(b,"spec_decode_num_accepted_tokens_total"),g(b,"spec_decode_num_draft_tokens_total")]
if None in p: print("  acceptance: n/a (MTP off)")
else:
    d=p[3]-p[1]
    print("  acceptance: %.1f%% (%d/%d)" % (100*(p[2]-p[0])/d, p[2]-p[0], d) if d>0 else "  acceptance: no drafts")
PY
  docker stop "$name" >/dev/null 2>&1 && echo "  stopped (kept)" | tee -a "$SUM"
  sleep 10
}

run_cfg A-nomtp-8192  ""  8192
run_cfg B-k3-8192     3   8192
run_cfg C-k3-32768    3   32768
run_cfg D-k3-16384    3   16384

echo "" | tee -a "$SUM"
echo "=== DONE $(date +%T) ===" | tee -a "$SUM"
echo "JC_V3_5090_DONE"
