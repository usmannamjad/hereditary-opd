#!/bin/bash
vllm serve google/gemma-3-27b-it \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8002