# #!/bin/bash
# vllm serve Qwen/Qwen3.5-2B-Base\
#     --dtype bfloat16 \
#     --max-model-len 8192 \
#     --port 8009

#!/bin/bash
vllm serve Qwen/Qwen3.5-2B\
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8015