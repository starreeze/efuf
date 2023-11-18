# -*- coding: utf-8 -*-
# @Date    : 2023-10-29 10:01:50
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

# from typing import Iterable
import torch, numpy


def to_device(batch, device="cuda", type_hint=None):
    if type_hint is None:
        type_hint = batch
    if batch is None:
        return None
    if isinstance(type_hint, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(type_hint, list):
        return [to_device(x, device) for x in batch]
    if isinstance(type_hint, tuple):
        return tuple(to_device(x, device) for x in batch)
    if isinstance(type_hint, torch.Tensor):
        return batch.to(device)
    if isinstance(type_hint, numpy.ndarray):
        return torch.tensor(batch, device=device)
    # if isinstance(type_hint, Iterable):
    #     return (to_device(x, device) for x in batch)
    raise NotImplementedError(f"Unknown type when casting to device: {type(type_hint).__name__}")
