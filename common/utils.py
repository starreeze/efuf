# -*- coding: utf-8 -*-
# @Date    : 2023-10-29 10:01:50
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

from typing import Callable
import torch, numpy
from torch.utils.data import DataLoader, Dataset


class NullableDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size, **kwargs):
        "a dataloader that allows zero batch size"
        self.null = batch_size == 0
        if not self.null:
            super().__init__(dataset, batch_size=batch_size, **kwargs)
            self.loader = iter(self)
        else:
            print("warning: batch size of the dataloader is set to 0")

    def __len__(self) -> int:
        if self.null:
            raise ValueError("Cannot get length of zero batch size dataloader")
        return super().__len__()

    def __iter__(self):
        return self

    def __next__(self):
        if self.null:
            return None
        return next(self.loader)


class ContinuousDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size, shuffle=True, drop_last=True, **kwargs):
        "a dataloader that never ends. drop_last is set to True"
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
        self.loader = iter(super().__iter__())

    def __len__(self) -> int:
        raise ValueError("Cannot get length of infinite dataloader")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(super().__iter__())
            return next(self.loader)


def create_placeholder(size_mb=30845, device="cuda"):
    return torch.empty([size_mb, 1024, 1024], device=device, dtype=torch.int8)


def to_device(batch, device="cuda"):
    if batch is None:
        return None
    try:
        return {k: to_device(v, device) for k, v in batch.items()}
    except AttributeError:
        pass
    if isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(x, device) for x in batch)  # type: ignore
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, numpy.ndarray):
        return torch.tensor(batch, device=device)
    return batch
    # raise NotImplementedError(f"Unknown type when casting to device: {type(batch).__name__}")


def concat_dict(*args):
    ret = {}
    for k in args[0].keys():
        if isinstance(args[0][k], torch.Tensor):
            ret[k] = torch.cat([x[k] for x in args])
        elif isinstance(args[0][k], list):
            ret[k] = sum([x[k] for x in args], [])
        else:
            raise NotImplementedError(f"Unknown type when concatenating: {type(args[0][k]).__name__}")
    return ret


class Logger:
    def __init__(self, log_fn: Callable[[dict[str, float]], None] = lambda _: None, name="", avg_prefix="avg"):
        self.logs: dict[str, list[float]] = {}
        self.log_fn = log_fn
        self.name = name
        self.avg_prefix = avg_prefix

    def get_average(self):
        return {f"{self.name}_{self.avg_prefix}_{k}": sum(v) / len(v) for k, v in self.logs.items()}

    def log(self, dry=False, **kwargs):
        for k, v in kwargs.items():
            if k not in self.logs:
                self.logs[k] = [v]
            else:
                self.logs[k].append(v)
        if not dry:
            self.log_fn({f"{self.name}_{k}": v[-1] for k, v in self.logs.items()})
            self.log_fn({f"{self.name}_{self.avg_prefix}_{k}": sum(v) / len(v) for k, v in self.logs.items()})

    def clear(self):
        self.logs = {}
