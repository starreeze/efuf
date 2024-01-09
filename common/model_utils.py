# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 18:38:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch
from minigpt4.common.eval_utils import init_model
from common.args import minigpt4_finetune_parser


def load_minigpt(ckpt, device="cuda", args: list[str] = []) -> tuple[torch.nn.Module, torch.nn.Module]:
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args(args), device)
    latest = ckpt if os.path.isfile(ckpt) else os.path.join(ckpt, sorted(os.listdir(ckpt))[-1])
    print(f"Loading from {latest}")
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)
    return model, vis_processor


def main():
    pass


if __name__ == "__main__":
    main()
