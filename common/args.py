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
parser.add_argument("--flattened_data_path", type=str, default="dataset/flattened.json")
## format
parser.add_argument("--column_splitter", type=str, default=" ### ")
parser.add_argument("--object_splitter", type=str, default=", ")
parser.add_argument("--subsentence_splitter_set", type=str, default=",.;!?")
parser.add_argument("--clip_prompt", type=str, default="A photo containing ")

# insight
## model
### llm for object extraction
parser.add_argument("--llama_path", type=str, default="/home/nfs02/model/llama2/hf/Llama-2-13b-chat-hf")
# parser.add_argument("--llama_batch_size", type=int, default=4)
### methods for insight
parser.add_argument("--patch_size", type=int, default=48)
parser.add_argument("--window_size", type=int, default=4)  # number of patches
parser.add_argument("--average_top_k", type=int, default=4)
## others
parser.add_argument("--least_data_size", type=int, default=50)
parser.add_argument("--sample_policy", type=str, default="random")

# finetune
parser.add_argument("--unlearn_target", type=str, default="subsentence", help="object or subsentence")
parser.add_argument("--hal_clip_thres", type=float, default=21, help="clip score < thres will be regraded as hal")
parser.add_argument("--norm_clip_thres", type=float, default=32, help="clip score > thres will be regraded as norm")
parser.add_argument("--max_new_tokens", type=int, default=200, help="max number of generated tokens")
parser.add_argument("--infer_sample_start", type=int, default=0)
parser.add_argument("--infer_sample_end", type=int, default=1000)
parser.add_argument("--infer_dataloader_worker", type=int, default=0)
parser.add_argument(
    "--infer_prompt",
    type=str,
    default="<Img><ImageHere></Img> Please describe the image in no more than 50 words. Make sure to be brief and concise.",
)
parser.add_argument("--train_dataloader_worker", type=int, default=0)
parser.add_argument("--train_prompt", type=str, default="<Img><ImageHere></Img> Please describe the image.")
## models
### minigpt
parser.add_argument(
    "--minigpt_cfg_path", default="configs/minigpt4_llama2_fp16.yaml", help="path to configuration file."
)
parser.add_argument("--minigpt_infer_batch_size", type=int, default=4)
parser.add_argument("--minigpt_infer_retry", type=int, default=3)
parser.add_argument("--minigpt_train_batch_size", type=int, default=4)

# common control
parser.add_argument("--restart", action="store_true")
parser.add_argument("--seed", type=int, default=28509)

args = parser.parse_args()
print(args)


# model provided parser
def minigpt4_finetune_parser():
    parser = argparse.ArgumentParser(description="finetune minigpt4")
    parser.add_argument("--cfg-path", default=args.minigpt_cfg_path, help="path to configuration file.")
    parser.add_argument("--name", type=str, default="A2", help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default="all", help="path to configuration file.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=args.max_new_tokens, help="max number of generated tokens"
    )
    parser.add_argument("--batch_size", type=int, default=args.minigpt_infer_batch_size)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser
