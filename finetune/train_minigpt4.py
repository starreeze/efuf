# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch, wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from minigpt4.common.eval_utils import init_model
from common.args import args, minigpt4_finetune_parser
from common.utils import to_device, Logger


class FinetuneDataset(Dataset):
    def __init__(self, vis_processor, score_filter=0):
        "score_filter: <0 negative, >0 positive, =0 all"
        super(FinetuneDataset, self).__init__()
        self.vis_processor = vis_processor
        print("constructing dataset ...")
        with open(args.flattened_data_path, "r") as f:
            data: list[dict[str, str | int]] = json.load(f)
        self.data = []
        for d in tqdm(data):
            # XXX here we use hard threshold
            if score_filter <= 0 and d["score"] < args.hal_clip_thres:
                d["score"] = -1
                self.data.append(d)
            elif score_filter >= 0 and d["score"] > args.norm_clip_thres:
                d["score"] = 1
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
            "instruction_input": f"{args.train_prompt} {context}",
            "answer": target,
            "score": sample["score"],
        }


class ContinuousDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size, shuffle=True, **kwargs):
        "a dataloader that never ends"
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.loader = iter(super(ContinuousDataLoader, self).__iter__())

    def __len__(self) -> int:
        raise ValueError("Cannot get length of infinite dataloader")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.loader = iter(super(ContinuousDataLoader, self).__iter__())
            batch = next(self.loader)
        return batch


def load_datasets(vis_processor, score_filter, train_bs, valid_bs, continuous=False):
    dataset = FinetuneDataset(vis_processor, score_filter)
    valid_size = int(args.valid_data_split * len(dataset))
    train, valid = random_split(dataset, [len(dataset) - valid_size, valid_size])
    if continuous:
        train_loader = ContinuousDataLoader(train, train_bs, shuffle=True)
    else:
        train_loader = DataLoader(train, train_bs, shuffle=True)
    valid_loader = DataLoader(valid, valid_bs, shuffle=False)
    return train_loader, valid_loader


def train_step(model: torch.nn.Module, pos, neg, step: int, epoch: int, optimizer: AdamW, logger: Logger):
    optimizer.zero_grad()
    loss_pos = model(to_device(pos), add_end_sym=False)["loss"]
    loss_neg = model(to_device(neg), add_end_sym=False)["loss"]
    loss = loss_pos - loss_neg
    logger.log(loss_pos=loss_pos.item(), loss_neg=loss_neg.item(), loss=loss.item())
    if step % args.print_per_n_step == 0:
        tqdm.write(
            f"epoch: {epoch+1}, step: {step+1}, loss_pos: {loss_pos.item()}, "
            f"loss_neg: {loss_neg.item()}, loss: {loss.item()}"
        )
    loss.backward()
    optimizer.step()


def evaluate(model: torch.nn.Module, loader_pos: DataLoader, loader_neg: DataLoader):
    print("evaluating...")
    logger = Logger(name="eval")
    with torch.no_grad():
        for batch_idx, (pos, neg) in tqdm(enumerate(zip(loader_pos, loader_neg))):
            loss_pos = model(to_device(pos), add_end_sym=False)["loss"]
            loss_neg = model(to_device(neg), add_end_sym=False)["loss"]
            loss = loss_pos - loss_neg
            logger.log(loss_pos=loss_pos.item(), loss_neg=loss_neg.item(), loss=loss.item())
    loss = logger.get_average()
    wandb.log(loss)
    print(f"eval loss: {loss}")


def train(
    model: torch.nn.Module,
    train_loader_pos: DataLoader[dict],
    valid_loader_pos: DataLoader[dict],
    train_loader_neg: DataLoader[dict],
    valid_loader_neg: DataLoader[dict],
    optimizer: AdamW,
):
    train_logger = Logger(wandb.log, "train")
    model.train()
    eval_per_n_step = len(train_loader_pos) // args.eval_per_epoch
    for epoch in range(args.minigpt_train_epoch):
        print(f"Epoch {epoch+1}:")
        for batch_idx, (pos, neg) in tqdm(enumerate(zip(train_loader_pos, train_loader_neg))):
            step = epoch * len(train_loader_neg) + batch_idx
            train_step(model, pos, neg, step, epoch, optimizer, train_logger)
            if step % eval_per_n_step == 0:
                evaluate(model, valid_loader_pos, valid_loader_neg)
        train_logger.clear()


def save_ckpt(model: torch.nn.Module):
    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient (vit, llama)
            del state_dict[k]
    save_path = args.minigpt_ckpt_path + ".pth"
    print(f"Saving checkpoint to {save_path} ...")
    torch.save(state_dict, save_path)


def main():
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args(["--cfg-path", args.minigpt_train_cfg]))
    train_pos, valid_pos = load_datasets(
        vis_processor, 1, args.minigpt_train_bs_pos, args.minigpt_infer_batch_size, continuous=True
    )
    train_neg, valid_neg = load_datasets(vis_processor, -1, args.minigpt_train_bs_neg, args.minigpt_infer_batch_size)
    optim = AdamW(model.parameters(), lr=args.minigpt_train_lr, weight_decay=args.minigpt_train_wd)

    os.environ["WANDB_API_KEY"] = args.wandb_key
    os.environ["WANDB_MODE"] = "online"
    wandb.init(project="lmm_hal", entity=args.wandb_user, name="minigpt", config=vars(args), sync_tensorboard=True)

    train(model, train_pos, train_neg, valid_pos, valid_neg, optim)
    evaluate(model, valid_pos, valid_neg)
    save_ckpt(model)


if __name__ == "__main__":
    main()
