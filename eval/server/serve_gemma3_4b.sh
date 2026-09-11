#!/bin/bash
vllm serve  google/gemma-3-4b-pt \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --port 8020