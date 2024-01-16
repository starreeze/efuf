# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 18:38:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch
from minigpt4.common.eval_utils import init_model
from lavis.models import load_model_and_preprocess
from common.args import minigpt4_finetune_parser


def load_ckpt(model, ckpt, device):
    latest = ckpt if os.path.isfile(ckpt) else os.path.join(ckpt, sorted(os.listdir(ckpt))[-1])
    print(f"Loading from {latest}")
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)


def load_minigpt(ckpt, device="cuda", args: list[str] = []) -> tuple[torch.nn.Module, torch.nn.Module]:
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args(args), device)
    load_ckpt(model, ckpt, device)
    return model, vis_processor


def load_blip(ckpt, device="cuda"):
    model, vis_processor, _ = load_model_and_preprocess("blip2_vicuna_instruct", "vicuna7b")
    load_ckpt(model, ckpt, device)
    return model, vis_processor


def main():
    pass


if __name__ == "__main__":
    main()
