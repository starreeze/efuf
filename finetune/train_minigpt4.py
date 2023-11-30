# -*- coding: utf-8 -*-
# @Date    : 2023-11-12 09:22:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from minigpt4.common.eval_utils import init_model
from common.args import args, minigpt4_finetune_parser
from insight.object import get_caption_info


class ImageObjectDataset(Dataset):
    def __init__(self, processor):
        super(ImageObjectDataset, self).__init__()
        self.processor = processor
        # as new generated data has no [], it is regarded as norm
        self.scores = np.load(args.norm_result_path)
        with open(args.object_result_path, "r") as f:
            self.objects = f.read().splitlines()
        with open(args.caption_data_path, "r") as f:
            self.captions = f.read().splitlines()
        assert len(self.scores) == len(self.objects) == len(self.captions)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        image_name, caption, _ = get_caption_info(self.captions[index])
        assert caption
        image_path = os.path.join(args.image_dir_path, image_name)
        img = self.processor(Image.open(image_path).convert("RGB"))


def main():
    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args([]))
    model.train()
    image_names = sorted(os.listdir(args.image_dir_path))[: args.infer_num_sample]
    dataloader = DataLoader(
        ImageObjectDataset(vis_processor), args.batch_size, False, num_workers=args.infer_dataloader_worker
    )


if __name__ == "__main__":
    main()
