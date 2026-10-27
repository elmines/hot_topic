#!/bin/bash

EXTRA_ARGS=($@)
model=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
API_PORT=${API_PORT:-8001}

$RUN_PREFIX env VLLM_SERVER_DEV_MODE=1 uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len $((1 << 14)) \
    --max-num-seqs 16 \
    --max-num-batched-tokens $((1 << 13)) \
    --gpu-memory-utilization 0.85 \
    "${EXTRA_ARGS[@]}"