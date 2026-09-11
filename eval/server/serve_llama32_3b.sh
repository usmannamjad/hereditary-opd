#!/bin/bash
vllm serve unsloth/Llama-3.2-3B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --port 8031