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
pred_path = os.path.join(args.qbench_result_path, f"{args.run_name}_{args.model}_{basename}.json")
question_path = os.path.join(args.qbench_data_path, "llvisionqa_dev.json")
image_dir = os.path.join(args.qbench_data_path, "images")
model_dtype = {"llava": torch.bfloat16, "llavarlhf": torch.bfloat16, "share4v": torch.bfloat16}


class QBenchData(Dataset):
    def __init__(self, processor, start=0, end=int(1e9)):
        super().__init__()
        self.processor = processor
        with open(question_path, "r") as f:
            questions: list[dict] = json.load(f)
        self.data = questions[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        question = self.data[index]
        image_path = os.path.join(image_dir, question["img_path"])
        image = Image.open(image_path).convert("RGB")
        answer = chr(ord("A") + question["candidates"].index(question["correct_ans"]))
        query = question["question"] + "\n"
        for label, candidate in zip(["A", "B", "C", "D"], question["candidates"]):
            query += f"{label}. {candidate}\n"
        text = getattr(args, f"{args.model}_eval_cqa_prompt").format(question=query)
        return self.processor(image), question["img_path"], text, answer

    def save(self):
        os.rename(question_path, question_path + ".bak")
        with open(question_path, "w") as f:
            json.dump(self.data, f)
        return self

    @staticmethod
    def restore():
        os.rename(question_path + ".bak", question_path)


class QBenchTrainData(QBenchData):
    def __getitem__(self, index):
        image, question_id, question, answer = super().__getitem__(index)
        return data_maps[args.model](dict(image=image, input=question, output=answer, score=0.0), add_end_sym=True)


def train(model, vis_processor, start, end):
    # we train both the models for fair comparison, making them better respond to short-answer vqa questions
    train_data = QBenchTrainData(vis_processor, start, end)
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
    eval_data = QBenchData(vis_processor, start, end)
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
                results.append({"question_id": q, "prediction": response})
                if ans in response:
                    correct += 1
            total += len(question_id)
    print(f"correct: {correct}, total: {total}, acc: {correct / total}")
    with open(pred_path, "w") as f:
        json.dump(results, f)


def eval():
    model, vis_processor = model_loaders[args.model](
        getattr(args, f"{args.model}_ckpt_load_path"),
        model_args=(["--cfg-path", args.minigpt_train_cfg] if args.model == "minigpt" else []),
    )
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    if not "skip_train" in args.run_name:
        train(model, vis_processor, start=args.default_eval_samples, end=args.end_pos)
    inference(model, vis_processor, start=0, end=args.default_eval_samples)
    # os.system(
    #     f"python sqa/calculate.py --base-dir {os.path.join(args.sqa_data_path, 'ScienceQA_DATA')} --result-file {pred_path}"
    # )


if __name__ == "__main__":
    eval()
