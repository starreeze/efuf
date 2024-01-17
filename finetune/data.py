# -*- coding: utf-8 -*-
# @Date    : 2024-01-15 19:33:09
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import ContinuousDataLoader
from common.args import args
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


input_dict_name_map = {
    "minigpt": {"image": "image", "input": "instruction_input", "output": "answer"},
    "blip": {"image": "image", "input": "text_input", "output": "text_output"},
}


def input_dict_to_model(input_dict):
    get_model_key = lambda key, model_dict: model_dict[key] if key in model_dict else key
    return {get_model_key(k, input_dict_name_map[args.model]): v for k, v in input_dict.items()}


class GoldData(Dataset):
    def __init__(self, vis_processor, model="minigpt"):
        super(GoldData, self).__init__()
        self.model_prompt = getattr(args, f"{model}_train_prompt")
        self.vis_processor = vis_processor
        with open(os.path.join(args.annotation_path, "captions_train2014.json"), "r") as f:
            self.data = json.load(f)["annotations"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_name = f"{args.image_prefix}{sample['image_id']:012d}.jpg"
        image_path = os.path.join(args.image_dir_path, image_name)
        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        caption = t if (t := sample["caption"]).endswith(tuple(args.subsentence_splitter_set)) else t + "."
        return input_dict_to_model(
            {"image": image, "input": self.model_prompt, "output": caption, "score": args.gold_clip_score}
        )


class PosNegData(Dataset):
    def __init__(self, vis_processor, score_filter=0, model="minigpt"):
        "score_filter: <0 negative, >0 positive, =0 all"
        super(PosNegData, self).__init__()
        self.model = model
        self.model_prompt = getattr(args, f"{model}_train_prompt")
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
        return input_dict_to_model(
            {
                "image": image,
                # this should be the full instruction, as prompt template in train cfg is {}
                "input": f"{self.model_prompt} {context}",
                "output": target,
                "score": sample["score"],
            },
        )


class SentenceData(Dataset):
    def __init__(self, vis_processor, model="minigpt"):
        super(SentenceData, self).__init__()
        self.model = model
        self.model_prompt = getattr(args, f"{model}_train_prompt")
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
        return input_dict_to_model(
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
    if continuous:
        train_loader = ContinuousDataLoader(train, train_bs, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train, train_bs, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, valid_bs, shuffle=False, drop_last=True)
    return train_loader, valid_loader
