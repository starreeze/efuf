# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 11:23:40
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

eval_data_path = os.path.abspath(os.path.join(args.pope_result_path, "data.json"))
basename = getattr(args, f"{args.model}_ckpt_load_path").split("/")[-1][:10]
pred_path = os.path.join(args.pope_result_path, f"{args.run_name}_{args.model}_{basename}.json")
model_dtype = {"llava": torch.bfloat16, "share4v": torch.bfloat16}


def construct_data():
    "construct test data for pope. This will only output the unused images as a test set"
    images = []
    with open(args.object_data_path, "r") as f:
        objects = f.read().splitlines()
    images_used = {obj.split(args.column_splitter)[0] for obj in objects}
    for name in filter(lambda x: x not in images_used, sorted(os.listdir(args.image_dir_path))):
        images.append({"image": os.path.abspath(os.path.join(args.image_dir_path, name))})
    os.makedirs(args.pope_result_path, exist_ok=True)
    with open(eval_data_path, "w") as f:
        json.dump(images, f, indent=2)
    print(
        "Construction done. Please run\n"
        f"cd evaluate/pope && python main.py --auto_seg True --dataset {os.path.basename(args.pope_result_path)} "
        f"--img_path {eval_data_path} --save_path . --seg_num 2000\n"
        "to finish construction"
    )


class PopeData(Dataset):
    def __init__(self, filename: str, processor, start, end):
        super().__init__()
        self.processor = processor
        with open(os.path.join(args.pope_result_path, filename), "r") as f:
            self.data = f.read().splitlines()[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = json.loads(self.data[index])
        image_path = os.path.join(args.image_dir_path, sample["image"])
        img = Image.open(image_path).convert("RGB")
        answer = "Yes." if sample["label"] == "yes" else "No."
        return dict(image=self.processor(img), input=sample["text"], output=answer)

    def save_to(self, filename: str):
        with open(os.path.join(args.pope_result_path, filename), "w") as f:
            f.write("\n".join(self.data))
        return self


class PopeTrainData(PopeData):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data["input"] = getattr(args, f"{args.model}_eval_pope_prompt").format(question=data["input"])
        return data_maps[args.model](data | {"score": 0.0}, add_end_sym=True)


def eval(input_filename):
    model, vis_processor = model_loaders[args.model](getattr(args, f"{args.model}_ckpt_load_path"))
    try:
        model.to(model_dtype[args.model])
    except KeyError:
        pass
    train_data = PopeTrainData(input_filename, vis_processor, 0, 1000)
    train_loader = DataLoader(train_data, args.train_bs_total, True, collate_fn=sample_collators[args.model])
    eval_filename = "eval_" + input_filename.split("_")[-1]
    eval_data = PopeData(input_filename, vis_processor, 1000, 2000).save_to(eval_filename)
    eval_loader = DataLoader(eval_data, args.infer_bs_total, False)

    # we train the model to make it better respond to yes/no questions
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
            batch: dict = to_device(batch)  # type: ignore
            texts = [getattr(args, f"{args.model}_eval_pope_prompt").format(question=q) for q in batch["input"]]
            with autocast(dtype=torch.float16):
                answers = generators[args.model](model, texts, batch["image"])
            for q, ans in zip(batch["input"], answers):
                results.append(json.dumps({"question": q, "answer": ans}))
    with open(pred_path, "w") as f:
        f.write("\n".join(results))
    print(
        "Inference done. Please run\n"
        f"python evaluate/pope/evaluate.py --label {os.path.join(args.pope_result_path, eval_filename)} --pred {pred_path}\n"
        "to continue"
    )


def main():
    if args.run_name == "construct":
        construct_data()
        return
    run, type = args.run_name.split("_")
    if run == "eval":
        eval(f"result_pope_{type}.json")
    else:
        raise ValueError("Invalid arguments")


if __name__ == "__main__":
    main()
