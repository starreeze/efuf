# -*- coding: utf-8 -*-
# @Date    : 2024-01-15 19:33:09
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"common data loader, without specific model configuration"

from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from common.args import args
from common.models import data_maps, sample_collators
from common.utils import ContinuousDataLoader


class GoldData(Dataset):
    def __init__(self, vis_processor):
        super(GoldData, self).__init__()
        self.model_prompt = getattr(args, f"{args.model}_train_vqa_prompt")
        self.vis_processor = vis_processor
        self.data = self.load_vqa()

    @staticmethod
    def load_vqa():
        if os.path.exists(args.vqa_data_path):
            processed = json.load(open(args.vqa_data_path, "r"))
            if len(processed) >= args.llava_data_size_k * 1000:
                return processed

        processed = []
        print("converting vqa data...")
        data = json.load(open(args.llava_data_path, "r"))
        for s in tqdm(data[: args.llava_data_size_k * 1000]):
            if "image" not in s:
                continue
            image_path = os.path.join(args.vqa_image_dir, s["image"])
            if not os.path.exists(image_path):
                continue
            processed.append(
                {
                    "input": s["conversations"][0]["value"].replace("<image>", "").strip(),
                    "output": s["conversations"][1]["value"].strip(),
                    "image": image_path,
                }
            )
        json.dump(processed, open(args.vqa_data_path, "w"), indent=2)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = self.vis_processor(Image.open(sample["image"]).convert("RGB"))
        return data_maps[args.model](
            {
                "image": image,
                "input": self.model_prompt.format(question=sample["input"]),
                "output": sample["output"],
                "score": args.gold_clip_score,
            }
        )


class PosNegData(Dataset):
    def __init__(self, vis_processor, score_filter):
        "score_filter: <0 negative, >0 positive, =0 all"
        super(PosNegData, self).__init__()
        self.model_prompt = getattr(args, f"{args.model}_train_prompt")
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
        return data_maps[args.model](
            {
                "image": image,
                # this should be the full instruction, as prompt template in train cfg is {}
                "input": f"{self.model_prompt} {context}",
                "output": target,
                "score": sample["score"],
            },
            add_end_sym=False,
        )


class SentenceData(Dataset):
    def __init__(self, vis_processor):
        super(SentenceData, self).__init__()
        self.model_prompt = getattr(args, f"{args.model}_train_prompt")
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
        return data_maps[args.model](
            {"image": image, "input": self.model_prompt, "output": sample["sentence"], "score": sample["mean"]},
        )


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
    loader_cls = ContinuousDataLoader if continuous else DataLoader
    train_loader = loader_cls(train, train_bs, shuffle=True, drop_last=True, collate_fn=sample_collators[args.model])
    valid_loader = loader_cls(valid, valid_bs, shuffle=False, drop_last=True, collate_fn=sample_collators[args.model])
    return train_loader, valid_loader
