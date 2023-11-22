# -*- coding: utf-8 -*-
# @Date    : 2023-10-29 10:01:50
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

# from typing import Iterable
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
    # if isinstance(type_hint, Iterable):
    #     return (to_device(x, device) for x in batch)
    raise NotImplementedError(f"Unknown type when casting to device: {type(batch).__name__}")
