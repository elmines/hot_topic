#!/bin/bash

EXTRA_ARGS=($@)
source "$(dirname "$0")/_utils.sh"
model=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
API_PORT=${API_PORT:-8001}

$RUN_PREFIX env uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len "$HT_MAX_MODEL_LEN" \
    --max-num-seqs "$HT_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$HT_MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization 0.85 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":true,"fuse_attn_quant":true,"eliminate_noops":true}}' \
    "${EXTRA_ARGS[@]}"
