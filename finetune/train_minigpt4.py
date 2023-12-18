# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from minigpt4.common.eval_utils import init_model
from common.args import args, minigpt4_finetune_parser


class FinetuneDataset(Dataset):
    def __init__(self, vis_processor, tokenizer):
        super(FinetuneDataset, self).__init__()
        self.vis_processor = vis_processor
        self.tokenizer = tokenizer  # prompt, image_list -> token ids
        print("constructing dataset ...")
        with open(args.flattened_data_path, "r") as f:
            data: list[dict[str, str | int]] = json.load(f)
        self.data = []
        for d in tqdm(data):
            # XXX here we use hard threshold
            if d["score"] < args.hal_clip_thres:
                d["score"] = -1
                self.data.append(d)
            elif d["score"] > args.norm_clip_thres:
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
        encoded_context = self.tokenizer(context, return_tensors="pt")["input_ids"]
        encoded_target = self.tokenizer(target, return_tensors="pt")["input_ids"][1:]
        target_mask = torch.cat(
            [torch.zeros(len(encoded_context), dtype=torch.int), torch.ones(len(encoded_target), dtype=torch.int)]
        )
        return image, torch.cat([encoded_context, encoded_target]), target_mask


def main():
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args([]))
    model.train()

    dataset = FinetuneDataset(vis_processor, model.get_context_emb)


if __name__ == "__main__":
    main()
