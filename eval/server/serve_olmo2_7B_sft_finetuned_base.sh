vllm serve /mnt/nfs-share/usman/hered/sft_baseline__allenai_OLMo-2-1124-7B__20260901_140737/merged \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name allenai/OLMo-2-1124-7B-SFT-Finetuned-Base \
    --port 8011