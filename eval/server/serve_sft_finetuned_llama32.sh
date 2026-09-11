#!/bin/bash
vllm serve  /mnt/nfs-share/usman/hered/sft_baseline__unsloth_Llama-3.2-3B__20260905_133010/merged/ \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --served-model-name unsloth/Llama-3.2-2B-sft-finetuned \
    --port 8032