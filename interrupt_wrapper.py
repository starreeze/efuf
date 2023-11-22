# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on a list to allow interruption and resume"""

from __future__ import annotations
import os, pickle, traceback
from typing import Any, Callable, Sequence


class ResumeWrapper:
    def __init__(self, data: Sequence, restart=False, bar=True, total_items=None, checkpoint_path="checkpoint.pkl"):
        self.data = data
        total_items = total_items if total_items is not None else len(data)
        self.checkpoint_path = checkpoint_path
        if restart:
            os.remove(checkpoint_path)
        try:
            with open(checkpoint_path, "rb") as checkpoint_file:
                checkpoint = pickle.load(checkpoint_file)
        except FileNotFoundError:
            checkpoint = 0
        if bar:
            from tqdm import tqdm

            self.wrapped_range = tqdm(range(checkpoint, total_items), initial=checkpoint, total=total_items)
        else:
            self.wrapped_range = range(checkpoint, total_items)
        self.wrapped_iter = iter(self.wrapped_range)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.checkpoint_path, "wb") as checkpoint_file:
            pickle.dump(self.index, checkpoint_file)
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(self.checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        return self.data[self.index]


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
        with open(checkpoint_path, "wb") as checkpoint_file:
            pickle.dump(i + 1, checkpoint_file)
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
