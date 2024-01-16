# -*- coding: utf-8 -*-
# @Date    : 2024-01-13 11:52:16
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys, os
from huggingface_hub import snapshot_download

model_name = sys.argv[1]


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    while True:
        try:
            snapshot_download(model_name, repo_type="model", resume_download=True)
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
