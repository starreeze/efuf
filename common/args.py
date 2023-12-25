# -*- coding: utf-8 -*-
# @Date    : 2023-10-26 19:51:58
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import argparse

parser = argparse.ArgumentParser()
# data
## path
parser.add_argument("--vqa_prompt_path", type=str, default="dataset/vqa_prompt.txt")
parser.add_argument("--vqa_data_path", type=str, default="dataset/rlhf.json")
parser.add_argument("--caption_prompt_path", type=str, default="dataset/caption_prompt_finetune.txt")
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
parser.add_argument("--subsentence_splitter_set", type=str, default=",.;!?:")
parser.add_argument("--clip_prompt", type=str, default="A photo containing ")

# insight
## model
### llm for object extraction
parser.add_argument("--llama_path", type=str, default="/home/nfs02/model/llama2/hf/Llama-2-7b-chat-hf")
parser.add_argument("--llama_8bit", action="store_true")
llama_instruction_placeholder = "$$$"
llama_sys_prompt = (
    "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
    "Strictly follow the instruction and always answer as helpfully as possible.\n"
    f"<</SYS>>\n\n</s> [INST] {llama_instruction_placeholder} [/INST] "
)
parser.add_argument("--llama_instruction_placeholder", type=str, default=llama_instruction_placeholder)
parser.add_argument("--llama_sys_prompt", type=str, default=llama_sys_prompt)
# parser.add_argument("--llama_batch_size", type=int, default=4)

parser.add_argument("--prompt_input_label", type=str, default="(Input)")
parser.add_argument("--prompt_output_label", type=str, default="(Output)")
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
parser.add_argument("--infer_dataloader_worker", type=int, default=0)
parser.add_argument(
    "--infer_prompt",
    type=str,
    default="<Img><ImageHere></Img> Please describe the image in no more than 50 words. Make sure to be brief and concise.",
)
parser.add_argument("--train_dataloader_worker", type=int, default=0)
# as context should not be counted in instruction, we need to remove prompt template from cfg and add it here
parser.add_argument(
    "--train_prompt", type=str, default="[INST] <Img><ImageHere></Img> Please describe the image. [/INST]"
)
parser.add_argument("--valid_data_split", type=float, default=0.1)
parser.add_argument("--wandb_user", type=str, default="starreeze")
parser.add_argument("--wandb_key", type=str, default="a48676b858238540d4fdf76b89d0366d611426f6")
parser.add_argument("--print_per_n_step", type=int, default=1)
parser.add_argument("--eval_per_epoch", type=int, default=5)
## models
### minigpt
parser.add_argument(
    "--minigpt_infer_cfg", default="configs/minigpt4_infer_fp16.yaml", help="path to configuration file."
)
parser.add_argument(
    "--minigpt_train_cfg", default="configs/minigpt4_train_fp16.yaml", help="path to configuration file."
)
parser.add_argument("--minigpt_infer_batch_size", type=int, default=8)
parser.add_argument("--minigpt_infer_retry", type=int, default=3)
parser.add_argument("--minigpt_train_bs_pos", type=int, default=2, help="number of positive samples in a batch")
parser.add_argument("--minigpt_train_bs_neg", type=int, default=2, help="number of negative samples in a batch")
parser.add_argument("--minigpt_train_lr", type=float, default=1e-5)
parser.add_argument("--minigpt_train_wd", type=float, default=0.05)
parser.add_argument("--minigpt_train_epoch", type=int, default=2)
parser.add_argument("--minigpt_ckpt_path", type=str, default="checkpoints/minigpt")

# common control
parser.add_argument("--restart", action="store_true")
parser.add_argument("--seed", type=int, default=28509)
parser.add_argument("--start_pos", type=int, default=0)
parser.add_argument("--end_pos", type=int, default=int(1e10))

args = parser.parse_args()
print(args)


# model provided parser
def minigpt4_finetune_parser():
    parser = argparse.ArgumentParser(description="finetune minigpt4")
    parser.add_argument("--cfg-path", default=args.minigpt_infer_cfg, help="path to configuration file.")
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
