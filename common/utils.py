# -*- coding: utf-8 -*-
# @Date    : 2023-10-29 10:01:50
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

from typing import Callable
import torch, numpy


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
        return tuple(to_device(x, device) for x in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, numpy.ndarray):
        return torch.tensor(batch, device=device)
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
