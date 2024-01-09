# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch, wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import Logger, ContinuousDataLoader, create_placeholder

placeholder = create_placeholder(45210)

from tqdm import tqdm
from time import time
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from common.model_utils import load_minigpt
from common.args import args

from finetune.loss import get_loss, WeightScheduler, get_loss_eval


class GoldData(Dataset):
    def __init__(self, vis_processor):
        super(GoldData, self).__init__()
        self.vis_processor = vis_processor
        with open(args.annotation_path, "r") as f:
            self.data = json.load(f)["annotations"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_name = f"{args.image_prefix}{sample['image_id']:012d}.jpg"
        image_path = os.path.join(args.image_dir_path, image_name)
        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        caption = t if (t := sample["caption"]).endswith(tuple(args.subsentence_splitter_set)) else t + "."
        return {
            "image": image,
            "instruction_input": args.minigpt_train_prompt,
            "answer": caption,
        }


class PosNegData(Dataset):
    def __init__(self, vis_processor, score_filter=0):
        "score_filter: <0 negative, >0 positive, =0 all"
        super(PosNegData, self).__init__()
        self.vis_processor = vis_processor
        print("constructing dataset ...")
        with open(args.pos_neg_data_path, "r") as f:
            data: list[dict[str, str | int]] = json.load(f)
        self.data = []
        for d in data:
            if score_filter <= 0 and d["score"] < args.hal_clip_thres:
                self.data.append(d)
            elif score_filter >= 0 and d["score"] > args.norm_clip_thres:
                self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(args.image_dir_path, sample["image"])
        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        pos = sample["position"]
        context, target = sample["sentence"][:pos], sample["sentence"][pos:]
        return {
            "image": image,
            # this should be the full instruction, as prompt template in train cfg is {}
            "instruction_input": f"{args.minigpt_train_prompt} {context}",
            "answer": target,
            "score": sample["score"],
        }


class SentenceData(Dataset):
    def __init__(self, vis_processor):
        super(SentenceData, self).__init__()
        self.vis_processor = vis_processor
        print("constructing dataset ...")
        with open(args.sentence_data_path, "r") as f:
            data: list[dict[str, str | float]] = json.load(f)
        self.data = [d for d in data if d["min"] > args.hal_clip_thres and d["mean"] > args.sentence_clip_thres]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(args.image_dir_path, sample["image"])
        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        return {
            "image": image,
            "instruction_input": args.minigpt_train_prompt,
            "answer": sample["sentence"],
            "score": sample["mean"],
        }


def load_datasets(vis_processor, train_bs, valid_bs, type, continuous=False):
    if type == "gold":
        dataset = GoldData(vis_processor)
    elif type == "sentence":
        dataset = SentenceData(vis_processor)
    elif type in [0, 1, -1]:
        dataset = PosNegData(vis_processor, type)
    else:
        raise ValueError("Unknown dataset type")
    valid_size = int(args.valid_data_split * len(dataset))
    train, valid = random_split(dataset, [len(dataset) - valid_size, valid_size])
    if continuous:
        train_loader = ContinuousDataLoader(train, train_bs, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train, train_bs, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, valid_bs, shuffle=False, drop_last=True)
    return train_loader, valid_loader


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
    for epoch in range(args.minigpt_train_epoch):
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
    save_path = os.path.join(args.minigpt_ckpt_save_path, f"step_{step:06d}.pth")
    print(f"Saving checkpoint to {save_path} ...")
    torch.save(state_dict, save_path)


def main():
    args.minigpt_ckpt_save_path = os.path.join(args.minigpt_ckpt_save_path, str(time()))
    os.makedirs(args.minigpt_ckpt_save_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["http_proxy"] = os.environ["https_proxy"] = args.proxy
    wandb.init(project="lmm_hal", entity=args.wandb_user, name="minigpt", config=vars(args), sync_tensorboard=False)
    print("W&B initialized.")

    model, vis_processor = load_minigpt(args.minigpt_ckpt_load_path, "cpu", ["--cfg-path", args.minigpt_train_cfg])
    global placeholder
    del placeholder
    model = model.to(args.device)
    print("resumed the checkpoint.")

    train_pos, valid_pos = load_datasets(
        vis_processor, args.minigpt_train_bs_pos, args.minigpt_infer_bs_pos, type=1, continuous=True
    )
    train_gold, valid_gold = load_datasets(
        vis_processor, args.minigpt_train_bs_gold, args.minigpt_infer_bs_gold, "gold", continuous=True
    )
    train_sent, valid_sent = load_datasets(
        vis_processor, args.minigpt_train_bs_sent, args.minigpt_infer_bs_sent, "sentence", continuous=True
    )
    train_neg, valid_neg = load_datasets(
        vis_processor, args.minigpt_train_bs_neg, args.minigpt_infer_bs_neg, type=-1, continuous=False
    )
    print("Datasets loaded.")

    optim = AdamW(model.parameters(), lr=args.minigpt_train_lr, weight_decay=args.minigpt_train_wd)

    train(model, train_pos, train_gold, train_sent, train_neg, valid_pos, valid_gold, valid_sent, valid_neg, optim)
    evaluate(model, valid_pos, valid_gold, valid_sent, valid_neg)
    save_ckpt(model, len(train_neg))


if __name__ == "__main__":
    main()
