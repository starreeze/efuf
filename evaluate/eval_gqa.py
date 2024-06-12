# -*- coding: utf-8 -*-
# @Date    : 2024-06-11 15:15:36
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, sys, json, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast  # type: ignore
from common.args import args
from common.models import model_loaders, model_forward, generators, data_maps, sample_collators
from common.utils import to_device

basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1][:10]
pred_path = os.path.join(args.gqa_data_path, "testdev_balanced_predictions.json")
train_question_path = os.path.join(args.gqa_data_path, "questions1.2", "val_balanced_questions.json")
eval_question_path = os.path.join(args.gqa_data_path, "questions1.2", "testdev_balanced_questions.json")
image_dir = os.path.join(args.gqa_data_path, "images")
model_dtype = {"llava": torch.bfloat16, "llavarlhf": torch.bfloat16, "share4v": torch.bfloat16}


class GQAData(Dataset):
    def __init__(self, processor, start=0, end=int(1e9)):
        super().__init__()
        self.processor = processor
        with open(eval_question_path, "r") as f:
            questions: dict = json.load(f)
        self.data = list(questions.items())[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        id, question = self.data[index]
        image_name = question["imageId"] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        answer = question["answer"]
        text = getattr(args, f"{args.model}_eval_vqa_prompt").format(question=question["question"])
        return self.processor(image), id, text, answer

    def save(self):
        os.rename(eval_question_path, eval_question_path + ".bak")
        with open(eval_question_path, "w") as f:
            json.dump({k: v for k, v in self.data}, f)
        return self

    @staticmethod
    def restore():
        os.rename(eval_question_path + ".bak", eval_question_path)


class GQATrainData(GQAData):
    def __init__(self, processor, start=0, end=int(1e9)):
        self.processor = processor
        with open(train_question_path, "r") as f:
            questions: dict = json.load(f)
        self.data = list(questions.items())[start:end]

    def __getitem__(self, index):
        image, question_id, question, answer = super().__getitem__(index)
        return data_maps[args.model](dict(image=image, input=question, output=answer, score=0.0), add_end_sym=True)


def train(model, vis_processor, start, end):
    # we train both the models for fair comparison, making them better respond to short-answer vqa questions
    train_data = GQATrainData(vis_processor, start, end)
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
    eval_data = GQAData(vis_processor, start, end)
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)
    model.eval()
    model.to(torch.float16)
    correct, total = 0, 0
    with torch.no_grad():
        results = []
        for batch in tqdm(eval_loader):
            image, question_id, question, answer = to_device(batch)  # type: ignore
            with autocast(dtype=torch.float16):
                responses = generators[args.model](model, question, image)
            for q, response, ans in zip(question_id, responses, answer):
                results.append({"questionId": q, "prediction": response.rstrip(".").lower()})
                if ans.lower() in response.lower():  # type: ignore
                    correct += 1
            total += len(question_id)
    with open(pred_path, "w") as f:
        json.dump(results, f)
    print(f"correct: {correct}, total: {total}, acc: {correct / total}")


def eval():
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    if not "skip_train" in args.run_name:
        train(model, vis_processor, start=0, end=args.end_pos)
    inference(model, vis_processor, start=0, end=args.default_eval_samples)
    # os.system(f"cd {args.gqa_data_path} && python eval/eval_1.py --tier testdev_balanced")


if __name__ == "__main__":
    eval()
