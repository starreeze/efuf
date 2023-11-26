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

# insight
## model
parser.add_argument("--patch_size", type=int, default=48)
parser.add_argument("--window_size", type=int, default=4)  # number of patches
parser.add_argument("--average_top_k", type=int, default=4)
## others
parser.add_argument("--least_data_size", type=int, default=50)
parser.add_argument("--sample_policy", type=str, default="random")

# finetune
parser.add_argument("--hal_clip_thres", type=float, default=21, help="clip score < thres will be regraded as hal")
parser.add_argument("--norm_clip_thres", type=float, default=32, help="clip score > thres will be regraded as norm")

# common control
parser.add_argument("--restart", action="store_true")
parser.add_argument("--seed", type=int, default=28509)

args = parser.parse_args()
