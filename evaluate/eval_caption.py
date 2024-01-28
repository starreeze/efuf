# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from io import TextIOWrapper
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from common.models import model_loaders, generators
from common.args import args


class CocoImageDataset(Dataset):
    def __init__(self, image_names: list[str], processor):
        super(CocoImageDataset, self).__init__()
        self.image_names = image_names
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(args.image_dir_path, self.image_names[index])
        img = self.processor(Image.open(image_path).convert("RGB"))
        if args.device != "cpu":
            img = img.to(torch.float16).to(args.device)
        return img, self.image_names[index]


def process_single(batch, model, prompt: str, output_fd: TextIOWrapper):
    images, image_names = batch
    texts = [prompt] * args.infer_bs_total
    results = [""] * args.infer_bs_total
    filtered = []
    for _ in range(args.infer_retry):
        with torch.no_grad():
            answers = generators[args.model](model, texts, images)
        for i, (name, answer) in enumerate(zip(image_names, answers)):
            if answer.replace("\n", "") and not results[i]:
                results[i] = name + " ### " + answer.replace("\n", " ")
        filtered = [r for r in results if r]
        if len(filtered) == len(results):
            break
    output_fd.write("\n".join(filtered) + "\n")
    output_fd.flush()


def main():
    model_load_path = getattr(args, f"{args.model}_ckpt_load_path")
    model_args = ["--cfg-path", args.minigpt_train_cfg] if args.model == "minigpt" else []
    model, vis_processor = model_loaders[args.model](model_load_path, args.device, False, model_args)
    with open(args.object_data_path, "r") as f:
        objects = f.read().splitlines()
    images_used = {obj.split(args.column_splitter)[0] for obj in objects}
    image_names = filter(lambda x: x not in images_used, sorted(os.listdir(args.image_dir_path)))
    image_names = list(image_names)[args.start_pos : args.end_pos]
    dataloader = DataLoader(
        CocoImageDataset(image_names, vis_processor),
        args.infer_bs_total,
        False,
        num_workers=args.infer_dataloader_worker,
    )

    prompt = getattr(args, f"{args.model}_eval_prompt")
    with open(args.caption_eval_path, "w" if args.restart else "a", encoding="utf-8") as f:
        for batch in tqdm(dataloader):
            process_single(batch, model, prompt, output_fd=f)


if __name__ == "__main__":
    main()
