# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, os, nltk, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.args import args
from evaluate.eval_utils import get_eval_caption
from tqdm import tqdm
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def bleu(hyps, refs):
    #  Calculate bleu_1 and bleu_2.
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        score = bleu_score.sentence_bleu(
            [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[1, 0, 0, 0]
        )
        bleu_1.append(score)
        score = bleu_score.sentence_bleu(
            [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[0.5, 0.5, 0, 0]
        )
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def main():
    image_ids, captions = get_eval_caption()

    with open(os.path.join(args.annotation_path, "captions_train2014.json"), "r") as f:
        data = json.load(f)
    ground_truth = []
    for image_id in tqdm(image_ids):
        for image in data["annotations"]:
            if image["image_id"] == image_id:
                ground_truth.append(image["caption"])
                break

    bleu_1, bleu_2 = bleu(captions, ground_truth)
    print(f"Bleu_1 Score: {bleu_1}" f"\nBleu_2 Score: {bleu_2}")


if __name__ == "__main__":
    main()
