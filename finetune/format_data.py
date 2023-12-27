# -*- coding: utf-8 -*-
# @Date    : 2023-11-29 18:07:24
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
from captions, objects and scores format a flattened json file,
where each object has an entry containing fields: sentence, sub-sentence/object position, score

normal text ... hallucination object/subsentence
                ^
                |
    sub-sentence/object position
    
sentence: type=str, stripped sentence until the target
sub-sentence/object mask: type=int, beginning position (char-level) of the unlearn target
score: float, clip score of the object
"""

from __future__ import annotations
import sys, os, json, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from tqdm import tqdm
import numpy as np
from common.args import args


def find_subsentence(sentence: str, target_word: str):
    sub_sentences = re.split(f"[{args.subsentence_splitter_set}]+", sentence)
    for sub_sentence in sub_sentences:
        if target_word in sub_sentence:
            start_position = sentence.find(sub_sentence)
            end_position = start_position + len(sub_sentence)
            return start_position, end_position
    tqdm.write(f"'{target_word}' not found in '{sentence}', skipping...")
    return None


def process_sample(sample: tuple[str, str, np.ndarray]) -> list[dict[str, str | int]] | None:
    caption, objects, scores = sample
    image, _, objects = objects.split(args.column_splitter)
    objects = [obj.strip("[]") for obj in objects.split(args.object_splitter)]
    if len(objects) != scores.shape[0]:
        tqdm.write("objects and scores not match! skipping...")
        return None

    results = []
    for object, score in zip(objects, scores):
        # XXX which to unlearn? objects, subsentence or else?
        if args.unlearn_target == "objects":
            index = caption.find(object)
            assert index != -1
            # XXX should we keep the former hallucination objects in the training sample?
            results.append(
                {
                    "image": image,
                    "sentence": caption[: index + len(object)],
                    "position": index,
                    "score": float(score),
                }
            )
        elif args.unlearn_target == "subsentence":
            result = find_subsentence(caption, object)
            if result is not None:
                start, end = result
                results.append({"image": image, "sentence": caption[:end], "position": start, "score": float(score)})
        else:
            raise NotImplementedError()
    return results


def main():
    with open(args.caption_data_path, "r") as f:
        captions = f.read().splitlines()
    captions_d = {}
    for sample in captions:
        name, caption = sample.split(args.column_splitter)
        captions_d[name] = caption
    with open(args.object_data_path, "r") as f:
        objects = f.read().splitlines()
    # as new generated data has no [], it is regarded as norm
    scores = np.load(args.norm_result_path, allow_pickle=True)
    assert len(objects) == len(scores)
    results = []
    for sample in tqdm(zip(objects, scores), total=len(objects)):
        object, score = sample
        caption = captions_d[object.split(args.column_splitter)[0]]
        result = process_sample((caption, object, score))
        if result is not None:
            results.extend(result)
    with open(args.flattened_data_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
