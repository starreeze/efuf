# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 18:38:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch


def load_ckpt(model, ckpt, device="cuda"):
    latest = ckpt if os.path.isfile(ckpt) else os.path.join(ckpt, sorted(os.listdir(ckpt))[-1])
    print(f"Loading from {latest}")
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)


def load_minigpt(ckpt, device="cuda", args: list[str] = []) -> tuple[torch.nn.Module, torch.nn.Module]:
    from common.args import minigpt4_finetune_parser
    from minigpt4.common.eval_utils import init_model

    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args(args), device)
    load_ckpt(model, ckpt, device)
    model.train()
    return model, vis_processor


def load_blip(ckpt, device="cuda"):
    from lavis.models import load_model_and_preprocess
    from lavis.common.registry import registry
    from lavis.models.blip2_models.blip2 import disabled_train

    registry.register_path("library_root", "lavis")
    model, vis_processor, _ = load_model_and_preprocess("blip2_vicuna_instruct", "vicuna7b")
    load_ckpt(model, ckpt, device)
    model.train()
    model.llm_model = model.llm_model.eval()
    for param in model.llm_model.parameters():
        param.requires_grad = False
    model.llm_model.train = disabled_train
    return model, vis_processor["train"]  # type: ignore


def main():
    pass


if __name__ == "__main__":
    main()
