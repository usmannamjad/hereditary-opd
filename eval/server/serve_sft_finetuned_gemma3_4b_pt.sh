#!/bin/bash
vllm serve  /mnt/nfs-share/usman/hered/sft_baseline__google_gemma-3-4b-pt__20260904_220014/merged/ \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --served-model-name google/gemma-3-4b-pt-sft-finetuned \
    --port 8021