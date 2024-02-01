# -*- coding: utf-8 -*

from __future__ import annotations
import sys, os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from evaluate.eval_utils import get_eval_caption
from collections import Counter
import numpy as np


def distinct(seqs):
    # Calculate intra distinct-1 and distinct-2.
    dist1, dist2 = [], []
    for seq in tqdm(seqs):
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        dist1.append(len(unigrams))
        dist2.append(len(bigrams) / 2)

    dist1 = np.average(dist1)
    dist2 = np.average(dist2)
    return dist1, dist2


def main():
    image_ids, captions = get_eval_caption()

    distinct1, distinct2 = distinct(captions)
    print(f"distinct-1: {distinct1}" f"\ndistinct-2: {distinct2}")


if __name__ == "__main__":
    main()
