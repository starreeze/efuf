# -*- coding: utf-8 -*-
# @Date    : 2023-12-26 13:15:02
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import torch, math
from common.utils import concat_dict, to_device
from common.args import args


class WeightScheduler:
    def __init__(
        self,
        start_weight: float,
        end_weight: float,
        total_step: int,
        start_step_pos=0.0,
        end_step_pos=1.0,
        type="linear",
    ):
        start_step, end_step = total_step * start_step_pos, total_step * end_step_pos
        self.start_step = start_step
        self.end_step = end_step
        self.start_weight = start_weight
        self.end_weight = end_weight
        if type == "linear":
            self.step_fn = lambda step: (step - start_step) / (end_step - start_step)
        elif type == "exp":
            self.step_fn = lambda step: 1 - math.exp(-(step - start_step) / (end_step - start_step))
        elif type == "const":
            assert start_weight == end_weight
            self.step_fn = lambda step: 0
        else:
            raise NotImplementedError()

    def at_step(self, step: int) -> float:
        if step < self.start_step:
            return self.start_weight
        elif step > self.end_step:
            return self.end_weight
        else:
            return self.start_weight + (self.end_weight - self.start_weight) * self.step_fn(step)


def batch_inference(model, end_sym: list[bool], *inputs, reduction="mean"):
    """
    Run batch inference (merge input) and return loss for better efficiency.
    And also, a strange bug in instruct-blip is that for every model call,
    the VRAM consumption will boost, so this serves a workaround.
    Note: we modified the model code to suit this method.
    """
    # batch size for different types of input
    bs = [input["image"].shape[0] for input in inputs]
    end_sym_per_sample = []
    for b, e in zip(bs, end_sym):
        end_sym_per_sample.extend([e] * b)

    model_inputs = concat_dict(*inputs)
    assert model_inputs["image"].shape[0] == len(end_sym_per_sample)
    loss = model(model_inputs, add_end_sym=end_sym_per_sample, reduction=reduction)["loss"]
    return torch.split(loss, loss.shape[0] // len(inputs))


def get_loss(
    model: torch.nn.Module,
    pos: dict,
    gold: dict,
    sent: dict,
    neg: dict,
    step: int,
    pos_w_scheduler: WeightScheduler,
    neg_w_scheduler: WeightScheduler,
):
    pos, gold, sent, neg = to_device([pos, gold, sent, neg], args.device)  # type: ignore
    loss_pos, loss_gold, loss_sent, loss_neg = batch_inference(
        model, [False, True, True, False], pos, gold, sent, neg, reduction="none"
    )
    # the higher the pos_score or the lower the neg_score, the stronger the influence
    max_score = args.hal_clip_thres + args.norm_clip_thres
    pos_w = pos_w_scheduler.at_step(step)
    neg_w = neg_w_scheduler.at_step(step)
    loss_pos_weighted = (loss_pos * pos["score"]).mean() * pos_w
    loss_neg_weighted = (loss_neg * (max_score - neg["score"])).mean() * neg_w
    loss_gold_weighted = loss_gold.mean() * args.gold_clip_score * args.gold_w
    loss_sent_weighted = (loss_sent * sent["score"]).mean() * args.sent_w

    loss = (loss_pos_weighted + loss_gold_weighted + loss_sent_weighted - loss_neg_weighted) / (
        pos_w + args.gold_w + args.sent_w - neg_w
    )
    return loss, loss_pos.mean(), loss_gold.mean(), loss_sent.mean(), loss_neg.mean()


def get_loss_eval(model: torch.nn.Module, pos: dict, gold: dict, sent: dict, neg: dict):
    "return the loss of the evaluation mode (only containing loss for different components, in dtype=float)"
    samples: list[dict] = to_device([pos, gold, sent, neg], args.device)  # type: ignore
    loss_pos, loss_gold, loss_sent, loss_neg = batch_inference(model, [False, True, True, False], *samples)
    return loss_pos.item(), loss_gold.item(), loss_sent.item(), loss_neg.item()


def main():
    pass


if __name__ == "__main__":
    main()
