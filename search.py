# -*- coding: utf-8 -*-
# @Date    : 2023-11-23 19:46:37
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from args import args
from clip import infer_object_image
from interrupt_wrapper import ResumeWrapper
import numpy as np


def main():
    patch_size_range = 16 * np.arange(1, 8)
    window_size_range = 2 * np.arange(1, 8)
    average_top_k_range = [1, 2, 4, 6, 8]
    args.restart = True
    with open("result.txt", "a") as f:
        for patch_size, window_size, average_top_k in ResumeWrapper(
            patch_size_range, window_size_range, average_top_k_range
        ):
            args.patch_size = patch_size
            args.window_size = window_size
            args.average_top_k = average_top_k
            hal, norm, std = infer_object_image(bar_position=1)
            if hal:
                f.write(
                    f"p{args.patch_size:03d}-w{args.window_size:02d}-a{args.average_top_k:02d}: "
                    f"{hal:.03f},{norm:.03f},{std:.03f},{(norm-hal)/std:.03f}\n"
                )
                f.flush()


if __name__ == "__main__":
    main()
