# -*- coding: utf-8 -*-
# @Date    : 2024-04-02 18:36:40
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
from common.utils import to_device, DirectDict as DD
from evaluate.vqa.vqa import VQA
from evaluate.vqa.evaluate import VQAEval

basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1][:10]
pred_path = os.path.join(args.vqa_result_path, f"{args.run_name}_{args.model}_{basename}.json")
model_dtype = {"llava": torch.bfloat16, "share4v": torch.bfloat16}


class VQAData(Dataset):
    def __init__(self, processor, start, end):
        super().__init__()
        self.processor = processor
        with open(args.vqa_question_path, "r") as f:
            questions = json.load(f)
        self.data = DD(questions | {"questions": questions["questions"][start:end]})
        with open(args.vqa_annotation_path, "r") as f:
            self.data.annotations = json.load(f)["annotations"][start:end]

    def __len__(self):
        return len(self.data.questions)

    def __getitem__(self, index: int):
        question = self.data.questions[index]
        image_name = f"{args.image_prefix}{question['image_id']:012}.jpg"
        image_path = os.path.join(args.image_dir_path, image_name)
        image = Image.open(image_path).convert("RGB")
        answer = self.data.annotations[index]["multiple_choice_answer"].capitalize() + "."
        text = getattr(args, f"{args.model}_eval_vqa_prompt").format(question=question["question"])
        return self.processor(image), question["question_id"], text, answer

    def save(self):
        path = os.path.join(args.vqa_result_path, "data_eval.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(self.data, f)
        return self


class VQATrainData(VQAData):
    def __getitem__(self, index):
        image, question_id, question, answer = super().__getitem__(index)
        return data_maps[args.model](dict(image=image, input=question, output=answer, score=0.0), add_end_sym=True)


def inference():
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    train_data = VQATrainData(vis_processor, 0, args.default_eval_samples)
    train_loader = DataLoader(train_data, args.train_bs_total, True, collate_fn=sample_collators[args.model])
    eval_data = VQAData(vis_processor, args.default_eval_samples, 2 * args.default_eval_samples).save()
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)

    # we train both the models for fair comparison, making them better respond to short-answer vqa questions
    optim = AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)
    for batch in tqdm(train_loader):
        optim.zero_grad()
        batch: dict = to_device(batch)  # type: ignore
        with autocast(dtype=torch.float16):
            loss = model_forward[args.model](model, batch, [True] * args.train_bs_total).mean()
        tqdm.write(f"loss: {loss.item()}")
        loss.backward()
        optim.step()

    model.eval()
    model.to(torch.float16)
    with torch.no_grad():
        results = []
        for batch in tqdm(eval_loader):
            image, question_id, question, answer = to_device(batch)  # type: ignore
            with autocast(dtype=torch.float16):
                answers = generators[args.model](model, question, image)
            for q, ans in zip(question_id, answers):
                results.append({"question_id": q.item(), "answer": ans})
    with open(pred_path, "w") as f:
        json.dump(results, f)


def eval():
    # if not os.path.exists(pred_path):
    inference()
    path = os.path.join(args.vqa_result_path, "data_eval.json")
    vqa = VQA(path, path)
    vqa_result = vqa.loadRes(pred_path, path)
    vqa_eval = VQAEval(vqa, vqa_result, n=2)
    vqa_eval.evaluate()
    print(f"\n{vqa_eval.accuracy['overall']: .02f}")


if __name__ == "__main__":
    eval()
