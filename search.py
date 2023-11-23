# -*- coding: utf-8 -*-
# @Date    : 2023-11-23 19:46:37
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from args import args
from clip import infer_object_image
import numpy as np


def main():
    patch_size_range = 16 * np.arange(1, 8)
    window_size_range = 2 * np.arange(1, 8)
    args.restart = True
    with open("result.txt", "w") as f:
        for patch_size in patch_size_range:
            args.patch_size = patch_size
            for window_size in window_size_range:
                args.window_size = window_size
                print(f"p{patch_size:03d}-w{window_size:03d}")
                hal, norm, std = infer_object_image()
                f.write(
                    f"p{patch_size:03d}-w{window_size:03d}: "
                    f"{hal:.03f},{norm:.03f},{std:.03f},{(norm-hal)/std:.03f}\n"
                )
                f.flush()


if __name__ == "__main__":
    main()
