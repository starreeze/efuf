# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 10:04:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""extract objects from the rlfh data (with hallucination only)"""

from __future__ import annotations
import json, re
from io import TextIOWrapper
from functools import partial
from gpt import anychat_gpt_35, g4f_gpt_4
from interrupt_wrapper import resumable_fn
from args import *


class GPTExtractor:
    def __init__(self, prompt_path, version=4) -> None:
        if version == 4:
            self.complete = partial(g4f_gpt_4, stream=False)
            self.backup = anychat_gpt_35
        elif version == 35:
            self.complete = anychat_gpt_35
            self.backup = partial(g4f_gpt_4, stream=False)
        else:
            raise NotImplementedError()
        with open(prompt_path, "r") as f:
            self.prompt = f.read()

    def extract(self, inputs: str) -> str:
        response = self.complete([{"role": "user", "content": self.prompt.format(inputs)}])  # type:ignore
        if "i'm sorry" in response.lower():  # type:ignore
            response = self.backup([{"role": "user", "content": self.prompt.format(inputs)}])  # type:ignore
        if "i'm sorry" in response.lower():  # type:ignore
            raise ValueError("gpt not willing to respond.")
        return response  # type:ignore


def extract_sample_vqa(sample: dict, extractor: GPTExtractor, output_fd: TextIOWrapper):
    if sample["hallucination"]:
        norm_index = sample["preference"]
        hal_index = 3 - norm_index
        hal_obj = extractor.extract(sample[f"output_{hal_index}"])
        norm_obj = extractor.extract(sample[f"output_{norm_index}"])
        output_fd.write(args.column_splitter.join([sample["image"], hal_obj, norm_obj]) + "\n")
        output_fd.flush()


def extract_sample_caption(sample: str, extractor: GPTExtractor, output_fd: TextIOWrapper):
    # image name ### caption */#
    image_name, caption = sample.split(args.column_splitter)
    if not caption:
        return

    rank = 0
    if caption[-1] == "*":
        caption = caption[:-1].strip()
        rank = 1
    elif caption[-1] == "#":
        caption = caption[:-1].strip()
        rank = -1

    objects_str = extractor.extract(caption).strip(',.:;"')
    objects = objects_str.split(args.object_splitter)
    objects_brackets = []
    for obj in objects:
        match = re.search(r"\[(\w )*" + obj + r"( \w)*\]", caption)
        objects_brackets.append(match.group(0) if match else obj)
    objects_str = args.object_splitter.join(objects_brackets)

    output_fd.write(args.column_splitter.join([image_name, str(rank), objects_str]) + "\n")
    output_fd.flush()


def extract_vqa():
    with open(args.vqa_data_path, "r") as f:
        vqa_data = json.load(f)
    extractor = GPTExtractor(args.vqa_prompt_path)
    with open(args.object_data_path, "a") as f:
        resumable_fn(partial(extract_sample_vqa, extractor=extractor, output_fd=f), vqa_data)


def extract_caption():
    with open(args.caption_data_path, "r") as f:
        captions = f.read().splitlines()
    extractor = GPTExtractor(args.caption_prompt_path)
    with open(args.object_data_path, "a") as f:
        resumable_fn(partial(extract_sample_caption, extractor=extractor, output_fd=f), captions)


if __name__ == "__main__":
    extract_caption()
    # print(GPTExtractor().extract(open("tmp.txt", "r").read()))
