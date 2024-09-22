# -*- coding: utf-8 -*-
# @Date    : 2024-07-27 02:09:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch, wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import Logger, to_device
from common.args import args
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast  # type: ignore
from torch.optim import AdamW
from common.models import model_loaders, model_forward
from finetune.data import load_datasets


def train_step(model: torch.nn.Module, inputs, step: int, epoch: int, optimizer: AdamW, logger: Logger):
    inputs = to_device(inputs, model.device)
    optimizer.zero_grad()
    with autocast(dtype=args.train_dtype):
        loss = torch.mean(model_forward[args.model](model, inputs, [True] * inputs["image"].shape[0]))
    logger.log(loss=loss.item())
    if step % args.print_per_n_step == 0:
        tqdm.write(f"epoch: {epoch+1}, step: {step+1}, loss: {loss.item()}")
    loss.backward()
    optimizer.step()


def evaluate(model: torch.nn.Module, loader):
    print("evaluating...")
    logger = Logger(name="eval")
    with torch.no_grad():
        for sample in tqdm(loader, total=len(loader)):
            sample = to_device(sample, model.device)
            with autocast(dtype=args.train_dtype):
                loss = torch.mean(model_forward[args.model](model, sample, [True] * sample["image"].shape[0]))
            logger.log(True, loss=loss)
    loss = logger.get_average()
    wandb.log(loss)
    print(f"eval loss: {loss}")


def train(model: torch.nn.Module, train_loader: DataLoader[dict], valid_loader: DataLoader[dict], optimizer: AdamW):
    train_logger = Logger(wandb.log, "train")
    # model.train()
    num_step = len(train_loader)
    eval_per_n_step = num_step // args.eval_per_epoch
    for epoch in range(args.train_epoch):
        print(f"Epoch {epoch+1}:")
        for batch_idx, sample in tqdm(enumerate(train_loader), total=num_step):
            step = epoch * num_step + batch_idx
            if step % eval_per_n_step == 0 and (step or not args.no_first_eval):
                evaluate(model, valid_loader)
                if step:
                    save_ckpt(model, step)
            train_step(model, sample, step, epoch, optimizer, train_logger)
        train_logger.clear()


def save_ckpt(model: torch.nn.Module, step: int):
    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient (vit, llama)
            del state_dict[k]
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    save_path = os.path.join(args.ckpt_save_path, f"step_{step:06d}.pth")
    print(f"Saving checkpoint to {save_path} ...")
    torch.save(state_dict, save_path)
    with open(os.path.join(args.ckpt_save_path, "config.json"), "w") as f:
        config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
        json.dump(config, f, indent=2)


def main():
    # print(args)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["http_proxy"] = os.environ["https_proxy"] = args.proxy
    wandb.init(project="lmm_hal", entity=args.wandb_user, name=args.model, config=vars(args), sync_tensorboard=False)
    print("W&B initialized.")

    model_load_path = getattr(args, f"{args.model}_ckpt_load_path")
    model_args = ["--cfg-path", args.minigpt_train_cfg] if args.model == "minigpt" else []
    model, vis_processor = model_loaders[args.model](model_load_path, args.device, True, model_args)
    # global placeholder
    # del placeholder
    print("resumed the checkpoint.")

    train_loader, valid_loader = load_datasets(
        vis_processor, args.train_bs_gold, args.infer_bs_gold, "gold", continuous=False
    )
    print("Datasets loaded.")

    optim = AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)

    if args.dry_run:
        exit(0)
    args.ckpt_save_path = os.path.join(getattr(args, f"{args.model}_ckpt_save_path"), args.run_name)
    if os.path.exists(args.ckpt_save_path):
        args.ckpt_save_path = os.path.join(getattr(args, f"{args.model}_ckpt_save_path"), str(time()))
        print(f"ckpt_save_path exists, saving ckpt to {args.ckpt_save_path}.")
    train(model, train_loader, valid_loader, optim)
    evaluate(model, valid_loader)
    save_ckpt(model, len(train_loader))


if __name__ == "__main__":
    main()
