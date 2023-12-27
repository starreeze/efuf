# -*- coding: utf-8 -*-
# @Date    : 2023-12-26 13:15:02
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import torch
from common.utils import to_device
from common.args import args


def get_loss(model: torch.nn.Module, pos: dict, neg: dict):
    pos, neg = to_device(pos), to_device(neg)  # type: ignore
    # or we can use kl divergence instead of contrastive
    loss_pos = model(pos, add_end_sym=False, reduction="none")["loss"]
    loss_neg = model(neg, add_end_sym=False, reduction="none")["loss"]

    # the higher the pos_score or the lower the neg_score, the stronger the influence
    max_score = args.hal_clip_thres + args.norm_clip_thres
    loss_pos_weighted = (loss_pos * pos["score"]).mean()
    loss_neg_weighted = (loss_neg * (max_score - neg["score"])).mean() * args.negative_weight

    loss = (loss_pos_weighted - loss_neg_weighted) / (1 + args.negative_weight)
    return loss, loss_pos.mean(), loss_neg.mean()


def main():
    pass


if __name__ == "__main__":
    main()
