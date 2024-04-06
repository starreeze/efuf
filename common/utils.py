# -*- coding: utf-8 -*-
# @Date    : 2023-10-29 10:01:50
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar
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


def to_device(batch, device="cuda", dtype=None):
    "dtype conversion for tensors will only be applied within the same basic type (e.g. int, float)"
    if batch is None:
        return None
    try:
        return {k: to_device(v, device, dtype) for k, v in batch.items()}
    except AttributeError:
        pass
    if isinstance(batch, list):
        return [to_device(x, device, dtype) for x in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(x, device, dtype) for x in batch)  # type: ignore
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
        if dtype is not None and (
            torch.is_floating_point(batch) == torch.is_floating_point(torch.tensor([], dtype=dtype))
        ):
            batch = batch.to(dtype)
        return batch
    if isinstance(batch, numpy.ndarray):
        batch = torch.tensor(batch, device=device)
        if dtype is not None:
            batch = batch.to(dtype)
        return batch
    return batch
    # raise NotImplementedError(f"Unknown type when casting to device: {type(batch).__name__}")


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


def merge_dict_set(x: dict[Any, set], y: dict[Any, set]) -> dict[Any, set]:
    ret = x.copy()
    for k, v in y.items():
        if k not in ret:
            ret[k] = set()
        ret[k].update(v)
    return ret


K = TypeVar("K")
V = TypeVar("V")


class DirectDict(dict[K, V], Generic[K, V]):
    def __getattr__(self, name: str) -> V:
        if name in dir(self):  # Check if it's a method/attribute of DirectDict or dict
            # Return the method/attribute from the class itself
            return object.__getattribute__(self, name)
        try:
            return self[name]  # type: ignore
        except KeyError:
            raise AttributeError(f"'DirectDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: V) -> None:
        if name in dir(dict) or name in self.__dict__:
            raise AttributeError(f"Setting the method/attribute '{name}' via attribute-style is not allowed")
        self[name] = value  # type: ignore

    def __delattr__(self, name: str) -> None:
        if name in dir(dict) or name in self.__dict__:
            raise AttributeError(f"Deleting the method/attribute '{name}' via attribute-style is not allowed")
        try:
            del self[name]  # type: ignore
        except KeyError:
            raise AttributeError(f"'DirectDict' object has no attribute '{name}'")
