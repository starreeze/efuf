# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on a list to allow interruption and resume"""

from __future__ import annotations
import os, traceback
from typing import Any, Callable, Iterable
from itertools import product


# wrap some iterables, no retrying and limited to sequence
class ResumeWrapper:
    def __init__(
        self,
        *data,
        mode="product",
        restart=False,
        bar=0,  # if -1 no bar at all
        total_items=None,
        checkpoint_path="checkpoint_ir",
        convert_type=list,
    ):
        if len(data) == 1:
            if convert_type is not None:
                self.data = convert_type(data[0])
            else:
                self.data = data[0]
        elif mode == "product":
            self.data = list(product(*data))
        elif mode == "zip":
            self.data = list(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")
        total_items = total_items if total_items is not None else len(self.data)  # type: ignore
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
        return self.data[self.index]  # type: ignore


# need hand-crafted function but support retrying and iterable
def resumable_fn(
    func: Callable[[Any], None],
    data: Iterable,
    restart=False,
    retry=5,
    on_error="raise",
    bar=True,
    total_items=None,
    checkpoint_path="checkpoint_ir",
) -> None:
    if total_items is None:
        total_items = len(data)  # type: ignore
    try:
        data[0]  # type: ignore
        iterator_mode = False
    except TypeError:
        data = iter(data)
        iterator_mode = True

    checkpoint = 0
    if restart:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    try:
        with open(checkpoint_path, "r") as checkpoint_file:
            checkpoint = int(checkpoint_file.read().strip())
    except FileNotFoundError:
        pass
    if bar:
        from tqdm import tqdm

        wrapped_range = tqdm(range(checkpoint, total_items), initial=checkpoint, total=total_items)
        checkpoint_range = tqdm(range(checkpoint))
    else:
        wrapped_range = range(checkpoint, total_items)
        checkpoint_range = range(checkpoint)

    if iterator_mode:
        print(f"skipping data until {checkpoint}")
        for i in checkpoint_range:
            next(data)  # type: ignore

    for i in wrapped_range:
        if iterator_mode:
            try:
                item = next(data)  # type: ignore
            except StopIteration:
                break
        else:
            item = data[i]  # type: ignore

        for j in range(retry):
            try:
                func(item)
            except KeyboardInterrupt:
                traceback.print_exc()
                exit(1)
            except Exception as e:
                if j == retry - 1:
                    if on_error == "raise":
                        print(f"All retry failed:")
                        traceback.print_exc()
                        return
                    elif on_error == "continue":
                        print(f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy.")
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
