# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch, wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import Logger, create_placeholder
from common.args import args

if args.device != "cpu":
    placeholder = create_placeholder(45210)
else:
    placeholder = None

from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from common.model_utils import load_minigpt, load_blip
from finetune.data import load_datasets
from finetune.loss import get_loss, WeightScheduler, get_loss_eval


def train_step(
    model: torch.nn.Module,
    pos,
    gold,
    sent,
    neg,
    step: int,
    epoch: int,
    pos_w_scheduler: WeightScheduler,
    neg_w_scheduler: WeightScheduler,
    optimizer: AdamW,
    logger: Logger,
):
    optimizer.zero_grad()
    loss, loss_pos, loss_gold, loss_sent, loss_neg = get_loss(
        model, pos, gold, sent, neg, step, pos_w_scheduler, neg_w_scheduler
    )
    logger.log(
        loss_pos=loss_pos.item(),
        loss_gold=loss_gold.item(),
        loss_sent=loss_sent.item(),
        loss_neg=loss_neg.item(),
        loss=loss.item(),
    )
    if step % args.print_per_n_step == 0:
        tqdm.write(
            f"epoch: {epoch+1}, step: {step+1}, loss_pos: {loss_pos.item()}, loss_gold: {loss_gold.item()}, "
            f"loss_sent: {loss_sent.item()}, loss_neg: {loss_neg.item()}, loss: {loss.item()}"
        )
    loss.backward()
    optimizer.step()


def evaluate(model: torch.nn.Module, loader_pos, loader_gold, loader_sent, loader_neg):
    print("evaluating...")
    logger = Logger(name="eval")
    with torch.no_grad():
        for pos, gold, sent, neg in tqdm(zip(loader_pos, loader_gold, loader_sent, loader_neg), total=len(loader_neg)):
            loss_pos, loss_gold, loss_sent, loss_neg = get_loss_eval(model, pos, gold, sent, neg)
            logger.log(True, loss_pos=loss_pos, loss_gold=loss_gold, loss_sent=loss_sent, loss_neg=loss_neg)
    loss = logger.get_average()
    wandb.log(loss)
    print(f"eval loss: {loss}")


def train(
    model: torch.nn.Module,
    train_loader_pos: DataLoader[dict],
    train_loader_gold: DataLoader[dict],
    train_loader_sent: DataLoader[dict],
    train_loader_neg: DataLoader[dict],
    valid_loader_pos: DataLoader[dict],
    valid_loader_gold: DataLoader[dict],
    valid_loader_sent: DataLoader[dict],
    valid_loader_neg: DataLoader[dict],
    optimizer: AdamW,
):
    train_logger = Logger(wandb.log, "train")
    model.train()
    num_step = len(train_loader_neg)
    eval_per_n_step = num_step // args.eval_per_epoch
    pos_w_scheduler = WeightScheduler(
        args.pos_w_start, args.pos_w_end, num_step, args.pos_w_start_step_pos, type=args.pos_w_sched_type
    )
    neg_w_scheduler = WeightScheduler(
        args.neg_w_start, args.neg_w_end, num_step, args.neg_w_start_step_pos, type=args.neg_w_sched_type
    )
    for epoch in range(args.train_epoch):
        print(f"Epoch {epoch+1}:")
        for batch_idx, (pos, gold, sent, neg) in tqdm(
            enumerate(zip(train_loader_pos, train_loader_gold, train_loader_sent, train_loader_neg)), total=num_step
        ):
            step = epoch * num_step + batch_idx
            if step % eval_per_n_step == 0:
                evaluate(model, valid_loader_pos, valid_loader_gold, valid_loader_sent, valid_loader_neg)
                save_ckpt(model, step)
            train_step(
                model, pos, gold, sent, neg, step, epoch, pos_w_scheduler, neg_w_scheduler, optimizer, train_logger
            )
        train_logger.clear()


def save_ckpt(model: torch.nn.Module, step: int):
    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient (vit, llama)
            del state_dict[k]
    ckpt_save_path = os.path.join(getattr(args, f"{args.model}_ckpt_save_path"), str(time()))
    os.makedirs(ckpt_save_path, exist_ok=True)
    save_path = os.path.join(ckpt_save_path, f"step_{step:06d}.pth")
    print(f"Saving checkpoint to {save_path} ...")
    torch.save(state_dict, save_path)
    with open(os.path.join(ckpt_save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def main():
    os.environ["WANDB_MODE"] = "offline"
    os.environ["http_proxy"] = os.environ["https_proxy"] = args.proxy
    wandb.init(project="lmm_hal", entity=args.wandb_user, name=args.model, config=vars(args), sync_tensorboard=False)
    print("W&B initialized.")

    if args.model == "minigpt":
        model, vis_processor = load_minigpt(args.minigpt_ckpt_load_path, "cpu", ["--cfg-path", args.minigpt_train_cfg])
    elif args.model == "blip":
        model, vis_processor = load_blip(args.blip_ckpt_load_path, "cpu")
    else:
        raise ValueError("Invalid model.")
    global placeholder
    del placeholder
    model = model.to(args.device)
    print("resumed the checkpoint.")

    train_pos, valid_pos = load_datasets(vis_processor, args.train_bs_pos, args.infer_bs_pos, type=1, continuous=True)
    train_gold, valid_gold = load_datasets(
        vis_processor, args.train_bs_gold, args.infer_bs_gold, "gold", continuous=True
    )
    train_sent, valid_sent = load_datasets(
        vis_processor, args.train_bs_sent, args.infer_bs_sent, "sentence", continuous=True
    )
    train_neg, valid_neg = load_datasets(vis_processor, args.train_bs_neg, args.infer_bs_eng, type=-1, continuous=False)
    print("Datasets loaded.")

    optim = AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)

    if args.dry_run:
        exit(0)
    train(model, train_pos, train_gold, train_sent, train_neg, valid_pos, valid_gold, valid_sent, valid_neg, optim)
    evaluate(model, valid_pos, valid_gold, valid_sent, valid_neg)
    save_ckpt(model, len(train_neg))


if __name__ == "__main__":
    main()
