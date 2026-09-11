#!/bin/bash
vllm serve allenai/OLMo-2-1124-7B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --port 8001