# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on a list to allow interruption and resume"""

from __future__ import annotations
import os, pickle, traceback
from typing import Any, Callable


def resumable(
    func: Callable[[Any], None], data: list, retry=5, bar=True, total_items=None, checkpoint_path="checkpoint.pkl"
) -> None:
    if total_items is None:
        total_items = len(data)
    checkpoint = 0
    try:
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
    except FileNotFoundError:
        pass
    if bar:
        from tqdm import tqdm

        wrapped_range = tqdm(range(checkpoint, total_items), initial=checkpoint, total=total_items)
    else:
        wrapped_range = range(checkpoint, total_items)
    for i in wrapped_range:
        item = data[i]
        for j in range(retry):
            try:
                func(item)
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                if j == retry - 1:
                    print(f"All retry failed:")
                    traceback.print_exc()
                    return
                print(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")
            else:
                break
        with open(checkpoint_path, "wb") as checkpoint_file:
            pickle.dump(i + 1, checkpoint_file)
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


def main():
    from time import sleep

    tmp = True

    def perform_operation(item):
        nonlocal tmp
        sleep(1)
        if item == 3:
            if tmp:
                tmp = False
                raise ValueError("here")

    data = list(range(10))
    resumable(perform_operation, data)


if __name__ == "__main__":
    main()
