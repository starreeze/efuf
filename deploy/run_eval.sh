#!/bin/bash
docker run --gpus all --ipc=host --network=host --rm -it \
    -v .:/workspace/hal \
    -v /data/cache/huggingface:/root/.cache/huggingface \
    -v /data/cache/torch:/root/.cache/torch \
    -v /root/nltk_data:/root/nltk_data \
    -v /data/NJU/lib/zf/LLaVA:/workspace/hal/LLaVA \
    mm_hal:1.3
