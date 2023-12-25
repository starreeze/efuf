# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from io import TextIOWrapper
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from minigpt4.common.eval_utils import init_model
from common.args import args, minigpt4_finetune_parser


class CocoImageDataset(Dataset):
    def __init__(self, image_names: list[str], processor):
        super(CocoImageDataset, self).__init__()
        self.image_names = image_names
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(args.image_dir_path, self.image_names[index])
        img = Image.open(image_path).convert("RGB")
        return self.processor(img), self.image_names[index]


def process_single(batch, model, output_fd: TextIOWrapper):
    images, image_names = batch
    texts = [args.infer_prompt] * args.minigpt_infer_batch_size
    results = [""] * args.minigpt_infer_batch_size
    filtered = []
    for _ in range(args.minigpt_infer_retry):
        answer = model.generate(images, texts, max_new_tokens=args.max_new_tokens)
        for i, (name, answer) in enumerate(zip(image_names, answer)):
            if answer.replace("\n", "") and not results[i]:
                results[i] = name + " ### " + answer.replace("\n", " ")
        filtered = [r for r in results if r]
        if len(filtered) == len(results):
            break
    output_fd.write("\n".join(filtered) + "\n")
    output_fd.flush()


def main():
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args([]))
    model.eval()
    image_names = sorted(os.listdir(args.image_dir_path))[args.start_pos : args.end_pos]
    dataloader = DataLoader(
        CocoImageDataset(image_names, vis_processor),
        args.minigpt_infer_batch_size,
        False,
        num_workers=args.infer_dataloader_worker,
    )
    with open(args.caption_data_path, "w" if args.restart else "a", encoding="utf-8") as f:
        for batch in tqdm(dataloader):
            process_single(batch, model=model, output_fd=f)


if __name__ == "__main__":
    main()
