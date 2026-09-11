vllm serve /mnt/nfs-share/usman/hered/sft_baseline__Qwen_Qwen3.5-2B-Base__20260902_115055/merged \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8010