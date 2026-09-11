#!/bin/bash
vllm serve google/gemma-4-31B-it \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --port 8028