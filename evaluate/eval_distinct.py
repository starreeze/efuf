# -*- coding: utf-8 -*

from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from common.utils import merge_dict_set
from nltk import ngrams
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def distinct(seqs):
    # Calculate intra distinct-1 and distinct-2. 
    batch_size = len(seqs)
    dist1, dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

    dist1 = np.average(dist1)
    dist2 = np.average(dist2)
    return dist1, dist2

def main():
    with open(args.caption_eval_path, "r") as f:
        content: list[str] = f.read().splitlines()
    image_ids, captions = [], []
    for line in content:
        try:
            image_name, caption = line.split(args.column_splitter)
        except ValueError as e:
            print(f"Skipping line {line} due to {e}")
            continue
        image_ids.append(int(image_name.split("_")[-1].split(".")[0]))
        captions.append(caption)

    distinct1, distinct2 = distinct(captions)
    print(
        f"distinct-1: {dist1}"
        f"\ndistinct-2: {dist2}"
    )


if __name__ == "__main__":
    main()
