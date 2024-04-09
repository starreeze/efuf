# -*- coding: utf-8 -*-
# @Date    : 2024-04-02 18:36:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast  # type: ignore
from common.args import args
from common.models import model_loaders, generators
from common.utils import to_device
from evaluate.eval_vqa import train

basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1][:10]
pred_name = f"{args.run_name}_" if args.run_name else ""
pred_path = os.path.join(args.mme_result_path, f"{pred_name}{args.model}_{basename}")
os.makedirs(pred_path, exist_ok=True)
model_dtype = {"llava": torch.bfloat16, "llavarlhf": torch.bfloat16, "share4v": torch.bfloat16}


class MMEData(Dataset):
    def __init__(self, processor, text_file: str):
        super().__init__()
        self.processor = processor
        image_dir = os.path.join(args.mme_image_path, text_file.rsplit(".")[0])
        self.image_dir = t if os.path.exists(t := os.path.join(image_dir, "images")) else image_dir
        with open(os.path.join(args.mme_text_path, text_file), "r") as f:
            filter_fn = lambda l: os.path.exists(os.path.join(self.image_dir, l.split("\t")[0]))
            self.data = list(filter(filter_fn, f.read().splitlines()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        line = self.data[index]
        image_name, question, answer = line.split("\t")
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        text = getattr(args, f"{args.model}_eval_mme_prompt").format(question=question)
        return self.processor(image), text, line


def inference(model, vis_processor, text_file):
    eval_data = MMEData(vis_processor, text_file)
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)
    model.eval()
    model.to(torch.float16)
    with torch.no_grad():
        results = []
        for batch in tqdm(eval_loader, position=1):
            images, questions, lines = to_device(batch)  # type: ignore
            with autocast(dtype=torch.float16):
                answers = generators[args.model](model, questions, images)
            for line, answer in zip(lines, answers):
                results.append(line + "\t" + answer.replace("\n", " "))
    with open(os.path.join(pred_path, text_file), "w") as f:
        f.write("\n".join(results))


def eval():
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    # if not os.path.exists(pred_path):
    if not "skip_train" in args.run_name:
        train(model, vis_processor, start=0, end=args.end_pos)
    for text_file in tqdm(os.listdir(args.mme_text_path), position=0):
        inference(model, vis_processor, text_file)
    print(
        "Inference done. Please run\n"
        f"python evaluate/mme/calculation.py --results_dir {pred_path}\n"
        "to finish evaluation"
    )


if __name__ == "__main__":
    eval()
