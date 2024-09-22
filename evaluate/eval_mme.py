# -*- coding: utf-8 -*-
# @Date    : 2024-04-02 18:36:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast  # type: ignore
from common.args import args
from common.models import model_loaders, generators
from common.utils import to_device
from evaluate.eval_vqa import train

basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1]
pred_name = f"{args.run_name}_" if args.run_name else ""
pred_path = os.path.join(args.mme_result_path, f"{pred_name}{args.model}_{basename}")
os.makedirs(pred_path, exist_ok=True)
# model_dtype = {"llava": torch.bfloat16, "llavarlhf": torch.bfloat16, "share4v": torch.bfloat16}
model_dtype = {"llavarlhf": torch.bfloat16}


class MMEData(Dataset):
    def __init__(self, processor, category_path: str):
        super().__init__()
        self.processor = processor

        if os.path.exists(os.path.join(category_path, "images")):
            image_path = os.path.join(category_path, "images")
            qa_path = os.path.join(category_path, "questions_answers_YN")
        else:
            image_path = qa_path = category_path
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path

        self.data = []
        for file in os.listdir(qa_path):
            if not file.endswith(".txt"):
                continue
            for line in open(os.path.join(qa_path, file), encoding="utf-8"):
                question, answer = line.strip().split("\t")
                image_globs = glob(os.path.join(image_path, file.split(".")[0] + ".*"))
                image = list(filter(lambda x: not x.endswith(".txt"), image_globs))
                if image:
                    self.data.append((image[0], question, answer))
                else:
                    tqdm.write("No image found for " + file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        image_path, question, answer = sample
        image = Image.open(image_path).convert("RGB")
        question = question.replace(" Please answer yes or no.", "")
        text = getattr(args, f"{args.model}_eval_vqa_prompt").format(question=question)
        return self.processor(image), text, "\t".join(sample)


def inference(model, vis_processor, category_dir):
    eval_data = MMEData(vis_processor, os.path.join(args.mme_data_path, category_dir))
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)
    model.eval()
    with torch.no_grad():
        results = []
        for batch in tqdm(eval_loader, position=1):
            images, questions, lines = to_device(batch, model_dtype.get(args.model, torch.float16))  # type: ignore
            # with autocast(dtype=torch.float16):
            answers = generators[args.model](model, questions, images)
            for line, answer in zip(lines, answers):
                results.append(line + "\t" + answer.replace("\n", " "))
    with open(os.path.join(pred_path, category_dir + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(results))


def eval():
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    # if not os.path.exists(pred_path):
    if not "skip_train" in args.run_name:
        train(model, vis_processor, start=0, end=args.end_pos)
    for category_dir in tqdm(os.listdir(args.mme_data_path), position=0):
        if os.path.isdir(os.path.join(args.mme_data_path, category_dir)):
            inference(model, vis_processor, category_dir)
    eval_cmd = f"python evaluate/mme/calculation.py --results_dir {pred_path}"
    print(f"Inference done. running `{eval_cmd}`")
    os.system(eval_cmd)


if __name__ == "__main__":
    eval()
