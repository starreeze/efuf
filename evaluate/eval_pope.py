# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 11:23:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from common.args import args
from common.models import load_minigpt
from common.utils import to_device

eval_data_path = os.path.abspath(os.path.join(args.pope_result_path, "data.json"))
minigpt_pred_path = os.path.join(args.pope_result_path, "minigpt.json")


def construct_data():
    "construct test data for pope. This will only output the unused images as a test set"
    images = []
    with open(args.object_data_path, "r") as f:
        objects = f.read().splitlines()
    images_used = {obj.split(args.column_splitter)[0] for obj in objects}
    for name in filter(lambda x: x not in images_used, sorted(os.listdir(args.image_dir_path))):
        images.append({"image": os.path.abspath(os.path.join(args.image_dir_path, name))})
    os.makedirs(args.pope_result_path, exist_ok=True)
    with open(eval_data_path, "w") as f:
        json.dump(images, f, indent=2)
    print(
        "Construction done. Please run\n"
        f"cd evaluate/pope && python main.py --auto_seg True --dataset {os.path.basename(args.pope_result_path)} "
        f"--img_path {eval_data_path} --save_path . --seg_num 1000\n"
        "to continue"
    )


class MiniGPTPopeData(Dataset):
    def __init__(self, filename: str, processor):
        super().__init__()
        self.processor = processor
        with open(os.path.join(args.pope_result_path, filename), "r") as f:
            self.data = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = json.loads(self.data[index])
        image_path = os.path.join(args.image_dir_path, sample["image"])
        img = Image.open(image_path).convert("RGB")
        return self.processor(img), sample["text"]


def eval_minigpt(filename="result_pope_random.json"):
    model, vis_processor = load_minigpt(args.minigpt_ckpt_save_path)
    model.eval()
    loader = DataLoader(MiniGPTPopeData(filename, vis_processor), args.infer_bs_total, False)
    with torch.no_grad():
        results = []
        for batch in tqdm(loader):
            images, questions = to_device(batch)  # type: ignore
            texts = [args.minigpt_eval_pope_prompt.format(question=q) for q in questions]
            answers = model.generate(images, texts, max_new_tokens=args.pope_max_new_tokens)
            for q, ans in zip(questions, answers):
                results.append(json.dumps({"question": q, "answer": ans}))
    with open(minigpt_pred_path, "w") as f:
        f.write("\n".join(results))
    print(
        "Inference done. Please run\n"
        f"python evaluate/pope/evaluate.py --label {os.path.join(args.pope_result_path, filename)} --pred {minigpt_pred_path}\n"
        "to continue"
    )


def main():
    op = input("(construct/eval)? ")
    if op == "construct":
        construct_data()
        return
    if op == "eval":
        model = input("which model? ")
        if model == "minigpt":
            eval_minigpt()
            return
    raise ValueError("Invalid arguments")


if __name__ == "__main__":
    main()
