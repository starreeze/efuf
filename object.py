# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 10:04:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""extract objects from the rlfh data (with hallucination only)"""

from __future__ import annotations
import json
from io import TextIOWrapper
from functools import partial
from gpt import anychat_gpt_35, g4f_gpt_4
from interrupt_wrapper import resumable

prompt_path = "object_prompt.txt"
rlhf_data_path = "dataset/rlhf.json"
object_data_path = "dataset/objects.txt"
splitter = "@@@@"


class GPTExtractor:
    def __init__(self, version=4, path=prompt_path) -> None:
        if version == 4:
            self.complete = partial(g4f_gpt_4, stream=False)
        elif version == 35:
            self.complete = anychat_gpt_35
        else:
            raise NotImplementedError()
        with open(path, "r") as f:
            self.prompt = f.read()

    def extract(self, inputs: str) -> str:
        return self.complete([{"role": "user", "content": self.prompt.format(inputs)}])  # type:ignore


def extract_sample(sample: dict, extractor: GPTExtractor, output_fd: TextIOWrapper):
    if sample["hallucination"]:
        norm_index = sample["preference"]
        hal_index = 3 - norm_index
        hal_obj = extractor.extract(sample[f"output_{hal_index}"])
        norm_obj = extractor.extract(sample[f"output_{norm_index}"])
        output_fd.write(splitter.join([sample["image"], hal_obj, norm_obj]) + "\n")
        output_fd.flush()


def main():
    with open(rlhf_data_path, "r") as f:
        rlhf_data = json.load(f)
    extractor = GPTExtractor()
    with open(object_data_path, "a") as f:
        resumable(partial(extract_sample, extractor=extractor, output_fd=f), rlhf_data)


if __name__ == "__main__":
    main()
    # print(GPTExtractor().extract(open("tmp.txt", "r").read()))
