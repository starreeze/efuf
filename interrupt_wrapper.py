# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on a list to allow interruption and resume"""

from __future__ import annotations
import os, traceback
from typing import Any, Callable, Iterable, Sequence
from itertools import product


# wrap some iterables, no retrying
class ResumeWrapper:
    def __init__(
        self,
        *data: Iterable,
        mode="product",
        restart=False,
        bar=0,  # if -1 no bar at all
        total_items=None,
        checkpoint_path="checkpoint_ir",
    ):
        if mode == "product":
            self.data = list(product(*data))
        elif mode == "zip":
            self.data = list(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")
        total_items = total_items if total_items is not None else len(self.data)
        self.checkpoint_path = checkpoint_path
        if restart:
            os.remove(checkpoint_path)
        try:
            with open(checkpoint_path, "r") as checkpoint_file:
                checkpoint = int(checkpoint_file.read().strip())
        except FileNotFoundError:
            checkpoint = 0
        if bar >= 0:
            from tqdm import tqdm

            self.wrapped_range = tqdm(
                range(checkpoint, total_items), initial=checkpoint, total=total_items, position=bar
            )
        else:
            self.wrapped_range = range(checkpoint, total_items)
        self.wrapped_iter = iter(self.wrapped_range)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(self.index))
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(self.checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        return self.data[self.index]


# need hand-crafted function but support retrying
def resumable_fn(
    func: Callable[[Any], None],
    data: Sequence,
    restart=False,
    retry=5,
    bar=True,
    total_items=None,
    checkpoint_path="checkpoint.pkl",
) -> None:
    if total_items is None:
        total_items = len(data)
    checkpoint = 0
    if restart:
        os.remove(checkpoint_path)
    try:
        with open(checkpoint_path, "r") as checkpoint_file:
            checkpoint = int(checkpoint_file.read().strip())
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
                traceback.print_exc()
                exit(1)
            except Exception as e:
                if j == retry - 1:
                    print(f"All retry failed:")
                    traceback.print_exc()
                    return
                print(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")
            else:
                break
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(i + 1))
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


def test_fn():
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
    resumable_fn(perform_operation, data)


def test_wrapper():
    from time import sleep

    for i in ResumeWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    test_wrapper()
