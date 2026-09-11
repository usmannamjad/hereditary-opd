#!/bin/bash
# vllm serve Qwen/Qwen3.5-0.8B-Base \
#     --dtype bfloat16 \
#     --max-model-len 8192 \
#     --port 8013

    #!/bin/bash
vllm serve Qwen/Qwen3.5-0.8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8014