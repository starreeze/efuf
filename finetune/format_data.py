# -*- coding: utf-8 -*-
# @Date    : 2023-11-29 18:07:24
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
from captions, objects and scores format a flattened json file,
where each object has an entry containing fields: sentence, sub-sentence mask, score
"""

from __future__ import annotations
import sys, os, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from common.args import args


def process_sample(sample: tuple[str, str, np.ndarray]):
    sentence, objects, scores = sample


def main():
    with open(args.caption_data_path, "r") as f:
        captions = f.read().splitlines()
    with open(args.object_data_path, "r") as f:
        objects = f.read().splitlines()
    scores = np.load(args.norm_result_path)  # as new generated data has no [], it is regarded as norm
    assert len(captions) == len(objects) == len(scores)
    data = list(zip(captions, objects, scores))
    with open(args.flattened_data_path, "w" if args.restart else "a") as f:
        pass


if __name__ == "__main__":
    main()
