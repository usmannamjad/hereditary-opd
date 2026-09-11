vllm serve /mnt/nfs-share/usman/vision-opd/model_checkpoints \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8092 \
    --trust-remote-code \
    --served-model-name vision-opd