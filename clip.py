# -*- coding: utf-8 -*-
# @Date    : 2023-02-02 10:22:49
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from args import *
from utils import to_device

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))


num_workers = 0
data_source = "caption-fine"


class ObjectData(Dataset):
    def __init__(self, processor, image_dir, object_path):
        super().__init__()
        self.processor = processor
        self.image_dir = image_dir
        with open(object_path, "r") as f:
            self.objects = f.read().splitlines()

    def __len__(self):
        return len(self.objects)

    @staticmethod
    def format_prompt(object_str: str) -> str:
        objects = object_str.split(object_splitter)
        if max([len(obj.split(" ")) for obj in objects]) > 10:
            return ""
        return clip_prompt + object_str.replace("[", "").replace("]", "") + "."

    def construct_data(self, image, text):
        if text:
            return self.processor(text=text, images=image, return_tensors="pt")
        return None

    @staticmethod
    def collate_fn(data: list):
        return data[0]


class VqaObjectData(ObjectData):
    """return: preprocessed pair of hal and norm"""

    def __init__(self, processor, image_dir, object_path):
        super().__init__(processor, image_dir, object_path)

    def __getitem__(self, idx: int):
        image, hal_obj, norm_obj = self.objects[idx].split(column_splitter)
        hal_obj, norm_obj = self.format_prompt(hal_obj), self.format_prompt(norm_obj)
        image = Image.open(os.path.join(image_dir_path, image_prefix + image))
        return self.construct_data(image, hal_obj), self.construct_data(image, norm_obj)


class CaptionCoarseObjectData(ObjectData):
    """return: preprocessed pair, if contains hal"""

    def __init__(self, processor, image_dir, object_path):
        super().__init__(processor, image_dir, object_path)

    def __getitem__(self, idx: int):
        image, rank, objects = self.objects[idx].split(column_splitter)
        hal = "[" in objects
        object_desc = self.format_prompt(objects)
        image = Image.open(os.path.join(image_dir_path, image))
        return self.construct_data(image, object_desc), hal


class CaptionFineObjectData(ObjectData):
    """return: preprocessed text and images, a boolean list indicating whether hal"""

    def __init__(self, processor, image_dir, object_path):
        super().__init__(processor, image_dir, object_path)

    def __getitem__(self, idx: int):
        image, rank, objects = self.objects[idx].split(column_splitter)
        objects = objects.split(object_splitter)
        hals = [obj[0] == "[" for obj in objects]
        object_descs = [self.format_prompt(obj.strip("[]")) for obj in objects]
        image = Image.open(os.path.join(image_dir_path, image))
        processed = self.processor(text=object_descs, images=image, return_tensors="pt", padding=True)
        return processed, hals


class ClipInfer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")  # type: ignore

    def process_sample(self, inputs):
        if inputs is None:
            return -1
        inputs = to_device(inputs)
        output = self.model(**inputs)["logits_per_image"][0]  # type: ignore
        if len(output) == 1:
            return float(output[0])
        return output

    def infer_vqa(self, dataset: VqaObjectData):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        hals, norms = [], []
        with torch.no_grad():
            for batch in tqdm(data):
                hals.append(self.process_sample(batch[0]))
                norms.append(self.process_sample(batch[1]))
        return hals, norms

    def infer_caption_coarse(self, dataset: CaptionCoarseObjectData):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        scores = ([], [])
        with torch.no_grad():
            for batch in tqdm(data):
                scores[1 - batch[1]].append(self.process_sample(batch[0]))
        return scores  # hals, norms

    def infer_caption_fine(self, dataset: CaptionFineObjectData):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        scores = ([], [])
        with torch.no_grad():
            for batch in tqdm(data):
                results = self.process_sample(batch[0])
                for score, hal in zip(results, batch[1]):  # type: ignore
                    scores[1 - hal].append(score)
        return to_device(scores, "cpu")  # hals, norms


def plot_histogram(hal, norm, filename="result.png", bins=np.arange(0, 1, 0.05)):
    hal = hal[hal != np.nan]
    norm = norm[norm != np.nan]
    plt.hist(hal, bins=bins, color="red", edgecolor="black", alpha=0.5)  # type: ignore
    plt.hist(norm, bins=bins, color="blue", edgecolor="black", alpha=0.5)  # type: ignore
    plt.savefig(filename)


def main():
    if not os.path.exists(hal_result_path):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        clip = ClipInfer()
        if data_source == "vqa":
            dataset = VqaObjectData(processor, image_dir_path, object_data_path)
            hals, norms = clip.infer_vqa(dataset)
        elif data_source == "caption-coarse":
            dataset = CaptionCoarseObjectData(processor, image_dir_path, object_data_path)
            hals, norms = clip.infer_caption_coarse(dataset)
        elif data_source == "caption-fine":
            dataset = CaptionFineObjectData(processor, image_dir_path, object_data_path)
            hals, norms = clip.infer_caption_fine(dataset)  # type: ignore
        else:
            raise NotImplementedError()
        np.save(hal_result_path, np.array(hals))
        np.save(norm_result_path, np.array(norms))

    hal, norm = np.load(hal_result_path), np.load(norm_result_path)
    hal[hal == -1] = np.nan
    norm[norm == -1] = np.nan
    normalize = lambda x, total: (x - np.nanmin(total)) / (np.nanmax(total) - np.nanmin(total))
    total = np.concatenate([hal, norm], axis=0)
    hal = normalize(hal, total)
    norm = normalize(norm, total)
    print(f"hal: mean {np.nanmean(hal)} std: {np.nanstd(hal)}")
    print(f"norm: mean {np.nanmean(norm)} std: {np.nanstd(norm)}")

    min_len = min(len(hal), len(norm))
    plot_histogram(hal[:min_len], norm[:min_len])


def test_run(image, text: str):
    image = Image.open(os.path.join(image_dir_path, image_prefix + image))
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    print(ClipInfer().process_sample(inputs))


if __name__ == "__main__":
    main()
