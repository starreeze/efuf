# -*- coding: utf-8 -*-
# @Date    : 2023-10-26 19:51:58
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import argparse

parser = argparse.ArgumentParser()
# data
## path
parser.add_argument("--vqa_prompt_path", type=str, default="dataset/vqa_prompt.txt")
parser.add_argument("--vqa_data_path", type=str, default="dataset/rlhf.json")
parser.add_argument("--caption_prompt_path", type=str, default="dataset/caption_prompt.txt")
parser.add_argument("--caption_data_path", type=str, default="dataset/captions.txt")
parser.add_argument("--object_data_path", type=str, default="dataset/objects.txt")
parser.add_argument("--image_dir_path", type=str, default="dataset/images")
parser.add_argument("--hal_result_path", type=str, default="dataset/hal.npy")
parser.add_argument("--image_prefix", type=str, default="COCO_train2014_")
parser.add_argument("--norm_result_path", type=str, default="dataset/norm.npy")
## format
parser.add_argument("--column_splitter", type=str, default=" ### ")
parser.add_argument("--object_splitter", type=str, default=", ")
parser.add_argument("--clip_prompt", type=str, default="A photo containing ")
## model
parser.add_argument("--patch_size", type=int, default=32)
parser.add_argument("--window_size", type=int, default=8)  # number of patches
parser.add_argument("--average_top_k", type=int, default=4)

parser.add_argument("--least_data_size", type=int, default=50)

# control
parser.add_argument("--restart", action="store_true")

args = parser.parse_args()
