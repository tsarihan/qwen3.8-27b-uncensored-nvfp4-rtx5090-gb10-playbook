#!/usr/bin/env bash
# Serve our own NVFP4A16 build of JonathanColetti/Qwen3.8-27B-Uncensored on ONE Spark (TP=1).
#
# Config is lifted from qwen3.8-27b-dgx-spark-gb10-playbook/scripts/serve-qwen38-nvfp4.sh,
# which is the proven recipe for this architecture on this silicon. Same image
# (vllm-dsv4:src-sm121), same sm_121a arch env, TP=1, GPU_UTIL 0.85, chunked prefill +
# async scheduling. No --quantization and no --kv-cache-dtype: the compressed-tensors
# config in config.json is authoritative and vLLM auto-routes it.
#
# DIFFERENCES from that playbook's model:
#   * this checkpoint is NVFP4A16 (weight-only, BF16 activations), not the unsloth
#     NVFP4+FP8 hybrid. 20.56 GB vs ~23.4 GB.
#   * the MTP head is a grafted BF16 shard (model-mtp.safetensors) carried in the index.
#
# Containers are NEVER removed here -- each boot gets a unique name so the MTP ladder can
# stop and start without deleting anything. Stop with `docker stop <name>`.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-Uncensored-JC-NVFP4A16}"
NAME="${NAME:-jc-nvfp4a16}"
PORT="${PORT:-8899}"
SERVED="${SERVED:-qwen3.8-27b-uncensored-jc}"
IMAGE="${IMAGE:-vllm-dsv4:src-sm121}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
SPEC_DECODE="${SPEC_DECODE:-off}"     # 'mtp' to enable the grafted draft head
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-}"
TOOLS="${TOOLS:-1}"                   # 0 to drop the tool-call parser
CACHE="${CACHE:-/data/models/vllm-cache-jc-nvfp4a16}"

mkdir -p "$CACHE" 2>/dev/null || CACHE="$HOME/vllm-cache-jc"; mkdir -p "$CACHE"

SPEC_ARGS=()
if [ "$SPEC_DECODE" != "off" ]; then
  J="{\"method\":\"${SPEC_DECODE}\""
  [ -n "$NUM_SPEC_TOKENS" ] && J="${J},\"num_speculative_tokens\":${NUM_SPEC_TOKENS}"
  J="${J}}"
  SPEC_ARGS=(--speculative-config "$J")
fi

TOOL_ARGS=()
[ "$TOOLS" = "1" ] && TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)

DOCKER="docker"; docker ps >/dev/null 2>&1 || DOCKER="sudo docker"

$DOCKER run -d --name "$NAME" --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/model:ro \
  -v "$CACHE":/cache \
  -e VLLM_CACHE_ROOT=/cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \
  "$IMAGE" \
  /model \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
    --enable-prompt-tokens-details \
    --reasoning-parser qwen3 \
    "${TOOL_ARGS[@]}" \
    "${SPEC_ARGS[@]}"

echo "[$NAME] TP=1 ctx=$MAX_MODEL_LEN util=$GPU_UTIL spec=$SPEC_DECODE k=${NUM_SPEC_TOKENS:-auto} port=$PORT"
