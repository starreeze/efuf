# -*- coding: utf-8 -*-
# @Date    : 2024-06-11 15:15:36
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from tqdm import tqdm
from typing import cast
from collections import Counter
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast  # type: ignore
from common.args import args
from common.models import model_loaders, model_forward, generators, data_maps, sample_collators
from common.utils import to_device

basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1][:10]
question_path = os.path.join(args.tqa_data_path, "llava_textvqa_val_v051_ocr.jsonl")
answer_path = os.path.join(args.tqa_data_path, "TextVQA_0.5.1_val.json")
image_dir = os.path.join(args.tqa_data_path, "train_images")
model_dtype = {"llava": torch.bfloat16, "llavarlhf": torch.bfloat16, "share4v": torch.bfloat16}


class TQAData(Dataset):
    def __init__(self, processor, start=0, end=int(1e9)):
        super().__init__()
        self.processor = processor
        with open(question_path, "r") as f:
            lines = f.read().strip().splitlines()[start:end]
        self.data: list[dict] = [json.loads(line) for line in lines]
        with open(answer_path, "r") as f:
            self.answer: list[dict] = json.load(f)["data"][start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        id, image, query, _ = self.data[index].values()
        image_path = os.path.join(image_dir, image)
        image = Image.open(image_path).convert("RGB")
        answers: list[str] = self.answer[index]["answers"]
        text = getattr(args, f"{args.model}_eval_tqa_prompt").format(question=query)
        return self.processor(image), id, text, answers

    def save(self):
        os.rename(question_path, question_path + ".bak")
        with open(question_path, "w") as f:
            json.dump({k: v for k, v in self.data}, f)
        return self

    @staticmethod
    def restore():
        os.rename(question_path + ".bak", question_path)


class TQATrainData(TQAData):
    def __getitem__(self, index):
        image, question_id, question, answers = super().__getitem__(index)
        answer = Counter(answers).most_common(1)[0][0]
        return data_maps[args.model](dict(image=image, input=question, output=answer, score=0.0), add_end_sym=True)


def train(model, vis_processor, start, end):
    # we train both the models for fair comparison, making them better respond to short-answer vqa questions
    train_data = TQATrainData(vis_processor, start, end)
    train_loader = DataLoader(train_data, args.train_bs_total, True, collate_fn=sample_collators[args.model])
    optim = AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)
    for batch in tqdm(train_loader):
        optim.zero_grad()
        batch: dict = to_device(batch)  # type: ignore
        with autocast(dtype=torch.float16):
            loss = model_forward[args.model](model, batch, [True] * args.train_bs_total).mean()
        tqdm.write(f"loss: {loss.item()}")
        loss.backward()
        optim.step()


def inference(model, vis_processor, start, end):
    eval_data = TQAData(vis_processor, start, end)
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)
    model.eval()
    model.to(torch.float16)
    correct, total = 0, 0
    with torch.no_grad():
        results = []
        for batch in tqdm(eval_loader):
            image, question_id, question, answers = to_device(batch)  # type: ignore
            answers = cast(list[list[str]], answers)
            with autocast(dtype=torch.float16):
                responses: list[str] = generators[args.model](model, question, image)
            for q, response, answer in zip(question_id, responses, answers):
                results.append({"questionId": q, "prediction": response})
                if any(ans.lower() in response.lower() for ans in answer):
                    correct += 1
            total += len(question_id)
    with open(args.tqa_result_path, "w") as f:
        json.dump(results, f)
    print(f"correct: {correct}, total: {total}, acc: {correct / total}")


def eval():
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    if not "skip_train" in args.run_name:
        train(model, vis_processor, start=args.default_eval_samples, end=args.end_pos)
    inference(model, vis_processor, start=0, end=args.default_eval_samples)


if __name__ == "__main__":
    eval()
