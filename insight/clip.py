# -*- coding: utf-8 -*-
# @Date    : 2023-11-22 10:22:49
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from abc import abstractmethod
import os, sys, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from transformers import CLIPProcessor, CLIPModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from common.utils import to_device

num_workers = 0
data_source = "caption-ftfi"


class ObjectData(Dataset):
    def __init__(self, processor, image_dir, object_path):
        super().__init__()
        self.processor = processor
        self.image_dir = image_dir
        with open(object_path, "r") as f:
            self.objects = f.read().splitlines()[args.start_pos : args.end_pos]

    def __len__(self):
        return len(self.objects)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @staticmethod
    def format_prompt(object_str: str) -> str:
        objects = object_str.split(args.object_splitter)
        if max([len(obj.split(" ")) for obj in objects]) > 10:
            return ""
        return args.clip_prompt + object_str.replace("[", "").replace("]", "") + "."

    def construct_single_pair(self, image, text):
        if text:
            return self.processor(text=text, images=image, return_tensors="pt")
        return None

    @staticmethod
    def collate_fn(data: list):
        return data[0]


class VqaObjectData(ObjectData):
    """return: preprocessed pair of hal and norm"""

    def __getitem__(self, idx: int):
        image, hal_obj, norm_obj = self.objects[idx].split(args.column_splitter)
        hal_obj, norm_obj = self.format_prompt(hal_obj), self.format_prompt(norm_obj)
        image = Image.open(os.path.join(args.image_dir_path, args.image_prefix + image)).convert("RGB")
        return self.construct_single_pair(image, hal_obj), self.construct_single_pair(image, norm_obj)


class CaptionCTCIData(ObjectData):  # coarse text coarse image
    """return: preprocessed pair, if contains hal"""

    def __getitem__(self, idx: int):
        image, rank, objects = self.objects[idx].split(args.column_splitter)
        hal = "[" in objects
        object_desc = self.format_prompt(objects)
        image = Image.open(os.path.join(args.image_dir_path, image)).convert("RGB")
        return self.construct_single_pair(image, object_desc), hal


class CaptionFTData(ObjectData):  # fine text
    def get_sample_data(self, idx: int):
        """return: image, object descriptions, hal"""
        image, rank, objects = self.objects[idx].split(args.column_splitter)
        objects = objects.split(args.object_splitter)
        hals = [bool(obj) and obj[0] == "[" for obj in objects]
        object_descs = [self.format_prompt(obj.strip("[]")) for obj in objects]
        image = Image.open(os.path.join(args.image_dir_path, image)).convert("RGB")
        return image, object_descs, hals


class CaptionFTCIData(CaptionFTData):  # fine text coarse image
    """return: preprocessed text and images, a boolean list indicating whether hal"""

    def __getitem__(self, idx: int):
        image, object_descs, hals = self.get_sample_data(idx)
        processed = self.processor(text=object_descs, images=image, return_tensors="pt", padding=True)
        return processed, hals


class CaptionFTFIData(CaptionFTData):  # fine text fine image
    """
    return: preprocessed text and images, a boolean list indicating whether hal,
    and a mask indicating how many times a patch is calculated
    """

    def __getitem__(self, idx: int):
        image, object_descs, hals = self.get_sample_data(idx)
        image = np.asarray(image)  # W, H, C
        images = []
        num_patches = (np.asarray(image.shape[:2]) + args.patch_size - 1) // args.patch_size
        patch_mask = torch.zeros(*num_patches, dtype=torch.int32)
        window_pixel = args.window_size * args.patch_size
        for w in range(num_patches[0] - args.window_size + 1):
            w_p = w * args.patch_size
            for h in range(num_patches[1] - args.window_size + 1):
                h_p = h * args.patch_size
                images.append(image[w_p : w_p + window_pixel, h_p : h_p + window_pixel])
                patch_mask[w : w + args.window_size, h : h + args.window_size] += 1
        if not images:
            return None
        processed = self.processor(text=object_descs, images=images, return_tensors="pt", padding=True)
        return processed, hals, patch_mask


class ClipInfer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")  # type: ignore

    def process_sample(self, inputs):
        if inputs is None:
            return -1
        inputs = to_device(inputs)
        output: torch.Tensor = self.model(**inputs)["logits_per_image"]  # type: ignore
        output = output.squeeze()
        if output.shape == ():
            return output.item()
        return output.to("cpu")

    def infer_vqa(self, dataset: VqaObjectData, bar_position=0):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        hals, norms = [], []
        with torch.no_grad():
            for batch in tqdm(data, position=bar_position):
                hals.append(self.process_sample(batch[0]))
                norms.append(self.process_sample(batch[1]))
        return hals, norms

    def infer_caption_CTCI(self, dataset: CaptionCTCIData, bar_position=0):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        scores = ([], [])
        with torch.no_grad():
            for batch in tqdm(data, position=bar_position):
                scores[1 - batch[1]].append(self.process_sample(batch[0]))
        return scores  # hals, norms

    def infer_caption_FTCI(self, dataset: CaptionFTCIData, bar_position=0):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        scores = ([], [])
        with torch.no_grad():
            for batch in tqdm(data, position=bar_position):
                results = self.process_sample(batch[0])
                for score, hal in zip(results, batch[1]):  # type: ignore
                    scores[1 - hal].append(score)
        return scores  # hals, norms

    def infer_caption_FTFI(self, dataset: CaptionFTFIData, bar_position=0):
        data = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)
        hals, norms = [], []
        with torch.no_grad():
            for batch in tqdm(data, position=bar_position):
                if batch is None:
                    continue
                mask: torch.Tensor = batch[2]
                # after reshape: [W, H, num_objects]
                results: torch.Tensor = self.process_sample(batch[0])  # type: ignore
                results = results.reshape(*(torch.tensor(mask.shape) - args.window_size + 1), -1)
                patch_score = torch.zeros(*mask.shape, results.shape[-1])
                for w in range(results.shape[0]):
                    for h in range(results.shape[1]):
                        patch_score[w : w + args.window_size, h : h + args.window_size] += results[w, h]
                patch_score = patch_score / mask.unsqueeze(2).expand(-1, -1, results.shape[-1])
                s = patch_score.shape
                patch_score = patch_score.reshape(s[0] * s[1], s[2]).transpose(0, 1)
                obj_score = torch.mean(patch_score.topk(args.average_top_k, dim=-1)[0], dim=-1)
                scores = ([], [])
                for score, hal in zip(obj_score, batch[1]):  # type: ignore
                    scores[1 - hal].append(float(score))
                hals.append(np.array(scores[0]))
                norms.append(np.array(scores[1]))
        return hals, norms


def plot_histogram(hal, norm, filename="result.png", bins=np.arange(15, 40, 1)):
    hal = hal[hal != np.nan]
    norm = norm[norm != np.nan]
    plt.hist(hal, bins=bins, color="red", edgecolor="black", alpha=0.5)  # type: ignore
    plt.hist(norm, bins=bins, color="blue", edgecolor="black", alpha=0.5)  # type: ignore
    plt.savefig(filename)
    plt.close()


def infer_object_image(bar_position=0, plot=True):
    if not os.path.exists(args.hal_result_path) or args.restart:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        clip = ClipInfer()
        if data_source == "vqa":
            dataset = VqaObjectData(processor, args.image_dir_path, args.object_data_path)
            hals, norms = clip.infer_vqa(dataset, bar_position)
        elif data_source == "caption-ctci":
            dataset = CaptionCTCIData(processor, args.image_dir_path, args.object_data_path)
            hals, norms = clip.infer_caption_CTCI(dataset, bar_position)
        elif data_source == "caption-ftci":
            dataset = CaptionFTCIData(processor, args.image_dir_path, args.object_data_path)
            hals, norms = clip.infer_caption_FTCI(dataset, bar_position)  # type: ignore
        elif data_source == "caption-ftfi":
            dataset = CaptionFTFIData(processor, args.image_dir_path, args.object_data_path)
            hals, norms = clip.infer_caption_FTFI(dataset, bar_position)  # type: ignore
        else:
            raise NotImplementedError()
        np.save(args.hal_result_path, np.array(hals, dtype="object"))
        np.save(args.norm_result_path, np.array(norms, dtype="object"))

    hal = np.concatenate(np.load(args.hal_result_path, allow_pickle=True))
    norm = np.concatenate(np.load(args.norm_result_path, allow_pickle=True))
    hal[hal == -1] = np.nan
    norm[norm == -1] = np.nan
    if hal.shape[0] < args.least_data_size or norm.shape[0] < args.least_data_size:
        tqdm.write("Not enough data to display")
        return 0, 0, 0, 0, 0
    hal_mean = np.nanmean(hal)
    hal_std = np.nanstd(hal)
    norm_mean = np.nanmean(norm)
    norm_std = np.nanstd(norm)
    all_std = np.nanstd(np.concatenate([hal, norm], axis=0))
    p_value_all = ttest_ind(hal, norm).pvalue

    identifier = f"p{args.patch_size:03d}-w{args.window_size:02d}-a{args.average_top_k:02d}"
    tqdm.write(identifier)
    tqdm.write(f"hal: mean {hal_mean} std: {hal_std}")
    tqdm.write(f"norm: mean {norm_mean} std: {norm_std}")
    tqdm.write(f"p-value-all: {p_value_all}")
    min_len = min(len(hal), len(norm))
    if args.sample_policy == "random":
        np.random.seed(args.seed)
        hal, norm = np.random.choice(hal, min_len), np.random.choice(norm, min_len)
    elif args.sample_policy == "max":
        hal = np.sort(hal)[:min_len]
        norm = np.sort(norm)[-min_len:]
    p_value_hist = ttest_ind(hal, norm).pvalue
    tqdm.write(f"p-value-hist: {p_value_hist}")
    if plot:
        plot_histogram(hal[:min_len], norm[:min_len], filename=identifier)
    return hal_mean, norm_mean, all_std, p_value_all, p_value_hist


def test_run(image, text: str):
    image = Image.open(os.path.join(args.image_dir_path, args.image_prefix + image))
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    print(ClipInfer().process_sample(inputs))


if __name__ == "__main__":
    infer_object_image()
