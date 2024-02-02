# -*- coding: utf-8 -*-
# @Date    : 2023-10-26 19:51:58
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import argparse, torch

parser = argparse.ArgumentParser()
# data
## path
parser.add_argument("--vqa_prompt_path", type=str, default="dataset/vqa_prompt.txt")
parser.add_argument("--vqa_data_path", type=str, default="dataset/rlhf.json")
parser.add_argument("--object_extract_prompt_path", type=str, default="dataset/caption_prompt_finetune.txt")
parser.add_argument(
    "--caption_data_path", type=str, default="dataset/captions.txt", help="captions generated by minigpt4 for training"
)
parser.add_argument(
    "--caption_eval_path",
    type=str,
    default="/workspace/hal/dataset/captions_eval.txt",
    help="captions generated by MLLM for evaluating",
)
parser.add_argument("--annotation_path", type=str, default="/workspace/hal/dataset/annotations")
parser.add_argument("--object_data_path", type=str, default="/workspace/hal/dataset/objects.txt")
parser.add_argument("--image_dir_path", type=str, default="/workspace/hal/dataset/images/images")
parser.add_argument("--hal_result_path", type=str, default="/workspace/hal/dataset/hal.npy")
parser.add_argument("--image_prefix", type=str, default="/workspace/hal/COCO_train2014_")
parser.add_argument("--norm_result_path", type=str, default="/workspace/hal/dataset/norm.npy")
parser.add_argument("--pos_neg_data_path", type=str, default="/workspace/hal/dataset/pos_neg.json")
parser.add_argument("--sentence_data_path", type=str, default="/workspace/hal/dataset/sentences.json")
parser.add_argument("--synonyms_path", type=str, default="/workspace/hal/dataset/synonyms.txt")
## format
parser.add_argument("--column_splitter", type=str, default=" ### ")
parser.add_argument("--object_splitter", type=str, default=", ")
parser.add_argument("--subsentence_splitter_set", type=str, default=",.;!?:")
parser.add_argument("--clip_prompt", type=str, default="A photo containing ")

# insight
## model
### llm for object extraction
parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-2-13b-chat-hf")
parser.add_argument("--llama_8bit", action="store_true")
llama_instruction_placeholder = "$$$"
llama_sys_prompt = (
    "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
    "Strictly follow the instruction and always answer as helpfully as possible.\n"
    f"<</SYS>>\n\n</s> [INST] {llama_instruction_placeholder} [/INST] "
)
parser.add_argument("--llama_instruction_placeholder", type=str, default=llama_instruction_placeholder)
parser.add_argument("--llama_sys_prompt", type=str, default=llama_sys_prompt)

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
parser.add_argument(
    "--hal_clip_thres", type=float, default=23, help="clip score < thres will be regraded as hal"
)  # 30k
parser.add_argument(
    "--norm_clip_thres", type=float, default=32, help="clip score > thres will be regraded as norm"
)  # 21k
parser.add_argument(
    "--sentence_clip_thres",
    type=float,
    default=27.5,
    help="sentences with mean clip score > thres will be used as whole-sentence positive sample",
)  # 27k
parser.add_argument("--gold_clip_score", type=float, default=40, help="clip score of the gold caption")

parser.add_argument("--neg_w_start", type=float, default=0.4)
parser.add_argument("--neg_w_end", type=float, default=0)
parser.add_argument("--neg_w_start_step_pos", type=float, default=0.4)
parser.add_argument("--neg_w_sched_type", type=str, default="linear")
parser.add_argument("--pos_w_start", type=float, default=1)
parser.add_argument("--pos_w_end", type=float, default=0.5)
parser.add_argument("--pos_w_start_step_pos", type=float, default=0)
parser.add_argument("--pos_w_sched_type", type=str, default="linear")
parser.add_argument("--gold_w", type=float, default=0)
parser.add_argument("--sent_w", type=float, default=0.2)

parser.add_argument("--max_new_tokens", type=int, default=200, help="max number of generated tokens")
parser.add_argument("--infer_dataloader_worker", type=int, default=0)
parser.add_argument("--valid_data_split", type=float, default=0.05)
parser.add_argument("--wandb_user", type=str, default="starreeze")
parser.add_argument("--print_per_n_step", type=int, default=5)
parser.add_argument("--eval_per_epoch", type=int, default=4)

## models
### common
parser.add_argument("--model", type=str, default="minigpt", help="model name to train")
parser.add_argument("--infer_bs_multiply", type=int, default=2)
parser.add_argument(
    "--train_bs_pos",
    type=int,
    default=1,
    help="number of positive samples (normal objects predicted by clip) in a batch",
)
parser.add_argument(
    "--train_bs_gold",
    type=int,
    default=1,
    help="number of positive samples (gold caption of COCO) in a batch",
)
parser.add_argument(
    "--train_bs_sent",
    type=int,
    default=1,
    help="number of positive samples (generated complete sentence) in a batch",
)
parser.add_argument("--train_bs_neg", type=int, default=1, help="number of negative samples in a batch")
parser.add_argument("--infer_bs_total", type=int, default=0, help="overwrite infer multiply for generatrion")
parser.add_argument("--train_lr", type=float, default=1e-5)
parser.add_argument("--train_wd", type=float, default=0.05)
parser.add_argument("--train_epoch", type=int, default=1)
parser.add_argument("--train_dataloader_worker", type=int, default=0)

### minigpt
parser.add_argument("--infer_retry", type=int, default=3)
parser.add_argument(
    "--minigpt_infer_cfg", default="configs/minigpt4_infer_fp16.yaml", help="path to configuration file."
)
parser.add_argument(
    "--minigpt_train_cfg", default="configs/minigpt4_train_fp16.yaml", help="path to configuration file."
)
parser.add_argument(
    "--minigpt_infer_prompt",
    type=str,
    default="<Img><ImageHere></Img> Please describe the image in no more than 50 words. Make sure to be brief and concise.",
)
# as context should not be counted in instruction, we need to remove prompt template from cfg and add it here
parser.add_argument(
    "--minigpt_train_prompt", type=str, default="[INST] <Img><ImageHere></Img> Please describe the image. [/INST]"
)
parser.add_argument(
    "--minigpt_eval_prompt",
    type=str,
    default="[INST] <Img><ImageHere></Img> Please describe the image in great detail. Your response should have at least 100 words. [/INST]",
)
parser.add_argument(
    "--minigpt_eval_pope_prompt",
    type=str,
    default="[INST] <Img><ImageHere></Img> According to the given image, answer yes or no to the question faithfully: {question} [/INST]",
)
parser.add_argument("--minigpt_ckpt_load_path", type=str, default="checkpoints/minigpt4_llama2_7b/pretrained.pth")
parser.add_argument("--minigpt_ckpt_save_path", type=str, default="checkpoints/minigpt4_llama2_7b")

### instruct-blip
parser.add_argument("--blip_train_prompt", type=str, default="Please describe the image.")
parser.add_argument(
    "--blip_eval_prompt",
    type=str,
    default="Please describe the image in great detail. Your response should have at least 100 words.",
)
# note that this should be modified in the config file, along with vicuna path
parser.add_argument("--blip_ckpt_load_path", type=str, default="checkpoints/blip_vicuna_7b/pretrained.pth")
parser.add_argument("--blip_ckpt_save_path", type=str, default="checkpoints/blip_vicuna_7b")

# mplug-owl: 
parser.add_argument("--owl_path", type=str, default="/workspace/hal/checkpoints/mplug_OWL_llama_7b")
parser.add_argument("--owl_ckpt_load_path", type=str, default="/workspace/hal/checkpoints/owl/1706760431.2430236") #todo tobe checked -> pass 

parser.add_argument("--owl_ckpt_save_path", type=str, default="/workspace/hal/checkpoints/owl") #todo: tobe checked -> pass 
parser.add_argument("--owl_use_bf16", type=bool, default=True)
parser.add_argument("--owl_train_prompt", type=str, default='''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please describe the image in great detail. Your response should have at least 100 words.
AI: ''')

parser.add_argument("--owl_eval_prompt", type=str, default='''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please describe the image in great detail. Your response should have at least 100 words.
AI: ''')

### llava
parser.add_argument(
    "--llava_train_prompt", type=str, default="### human: <image>\n Please describe the image. \n### gpt:"
)
parser.add_argument(
    "--llava_eval_prompt",
    type=str,
    default="### human: <image>\n Please describe the image in great detail. Your response should have at least 100 words. \n### gpt:",
)
parser.add_argument(
    "--llava_ckpt_load_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965",
)
parser.add_argument(
    "--llava_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965",
)
parser.add_argument(
    "--llava_vit_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
)
parser.add_argument("--llava_ckpt_save_path", type=str, default="checkpoints/llava_vicuna_7b")

### share4v
parser.add_argument(
    "--share4v_train_prompt", type=str, default="### human: <image>\n Please describe the image. \n### gpt:"
)
parser.add_argument(
    "--share4v_eval_prompt",
    type=str,
    default="### human: <image>\n Please describe the image in great detail. Your response should have at least 100 words. \n### gpt:",
)
parser.add_argument(
    "--share4v_ckpt_load_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B/snapshots/a973da7d8dba5e9ac2817f1c88bf9c8f36004078",
)
parser.add_argument(
    "--share4v_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B/snapshots/a973da7d8dba5e9ac2817f1c88bf9c8f36004078",
)
parser.add_argument(
    "--share4v_vit_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B_Pretrained_vit-large336-l12/snapshots/55da275fb4755cc5e5d9c6121aa72adc6de01f55",
)
parser.add_argument("--share4v_ckpt_save_path", type=str, default="checkpoints/share4v_7b")

# eval
parser.add_argument("--pope_result_path", type=str, default="evaluate/pope/result")
parser.add_argument("--pope_max_new_tokens", type=int, default=20)
parser.add_argument("--default_eval_samples", type=int, default=1600)

# common control
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--restart", action="store_true")
parser.add_argument("--seed", type=int, default=28509)
parser.add_argument("--start_pos", type=int, default=0)
parser.add_argument("--end_pos", type=int, default=int(1e10))
parser.add_argument("--proxy", type=str, default="")
parser.add_argument("--train_dtype_str", type=str, default="bfloat16")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--no_first_eval", action="store_true")

args = parser.parse_args()
args.infer_bs_pos = args.train_bs_pos * args.infer_bs_multiply
args.infer_bs_sent = args.train_bs_sent * args.infer_bs_multiply
args.infer_bs_eng = args.train_bs_neg * args.infer_bs_multiply
args.infer_bs_gold = args.train_bs_gold * args.infer_bs_multiply
if args.infer_bs_total == 0:
    args.infer_bs_total = args.infer_bs_pos + args.infer_bs_sent + args.infer_bs_eng + args.infer_bs_gold

args.train_dtype = getattr(torch, args.train_dtype_str)


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
    parser.add_argument("--batch_size", type=int, default=1)
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
