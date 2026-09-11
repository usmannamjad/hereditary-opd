vllm serve /mnt/nfs-share/usman/hered/sft_baseline__allenai_OLMo-2-1124-7B__20260908_044907/merged \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name allenai/OLMo-2-1124-7B-SFT-Finetuned-prompt-2 \
    --port 8080