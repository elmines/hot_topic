#!/bin/bash

set -x
# Need that dev mode to get the /server_info endpoint to get the maximum model length
# model=openai/gpt-oss-120b
model=openai/gpt-oss-20b

VLLM_SERVER_DEV_MODE=1 uv run vllm serve \
    $model \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key derf \
    --max-model-len $((1 << 14)) \
    --max-num-seqs 1 \
    --max-num-batched-tokens $((1 << 13)) \
    --gpu-memory-utilization 0.85