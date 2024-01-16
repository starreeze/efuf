# -*- coding: utf-8 -*-
# @Date    : 2024-01-14 09:42:57
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys, os


def caption():
    ckpt_dir = sys.argv[2]
    output_path = "eval_caption"
    for file in os.listdir(ckpt_dir):
        ckpt = os.path.join(ckpt_dir, file)
        os.system(
            f"python evaluate/eval_caption_minigpt4.py --minigpt_ckpt_save_path {ckpt} "
            f"--caption_eval_path {output_path}_{file}.txt --no_print_args"
        )


def chair():
    caption_dir = sys.argv[2]
    for file in os.listdir(caption_dir):
        if file.startswith("eval_caption"):
            caption = os.path.join(caption_dir, file)
            print(f"evaluating: {file}")
            os.system(f"python evaluate/eval_chair.py --caption_eval_path {caption} --no_print_args")


if __name__ == "__main__":
    if sys.argv[1] == "caption":
        caption()
    elif sys.argv[1] == "chair":
        chair()
    else:
        raise ValueError("Invalid argument")
