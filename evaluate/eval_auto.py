# -*- coding: utf-8 -*-
# @Date    : 2024-01-10 11:15:40
# @Author  : Shangyu.Xing (starreeze@foxmail.com); Weihao.Chen
# @Note    : some part of the code is from RLHF-V

from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
import numpy as np
from tqdm import tqdm
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from common.utils import merge_dict_set
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction


def bleu(hyps, refs):
    #  Calculate bleu_1 and bleu_2.
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in tqdm(zip(hyps, refs), total=len(hyps)):
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


def distinct(seqs):
    # Calculate intra distinct-1 and distinct-2.
    dist1, dist2 = [], []
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        dist1.append(len(unigrams))
        dist2.append(len(bigrams) / 2)

    dist1 = np.average(dist1)
    dist2 = np.average(dist2)
    return dist1, dist2


class CHAIR(object):
    lemma = nltk.wordnet.WordNetLemmatizer()  # type: ignore

    def __init__(self, annotation_path: str, synonyms_path: str, image_ids: Iterable[int]):
        self.annotation_path = annotation_path
        self.synonyms_path = synonyms_path
        self.image_ids = set(image_ids)
        self.load_mappings()
        # construct a set of gold object (category) from caption and instance annotation
        self.gold_objects = merge_dict_set(self.get_gold_caption_object(), self.get_gold_instance_object())

    def get_gold_instance_object(self) -> dict[int, set[str]]:
        with open(os.path.join(self.annotation_path, "instances_train2014.json"), "r") as f:
            content = json.load(f)
            instances = content["annotations"]
            labels = content["categories"]
        category_id_to_name = {l["id"]: l["name"] for l in labels}
        image_id_to_objects: dict[int, set[str]] = {
            i["image_id"]: set() for i in instances if i["image_id"] in self.image_ids
        }
        for i in instances:
            if i["image_id"] in self.image_ids:
                image_id_to_objects[i["image_id"]].add(category_id_to_name[i["category_id"]])
        return image_id_to_objects

    def get_gold_caption_object(self) -> dict[int, set[str]]:
        with open(os.path.join(self.annotation_path, "captions_train2014.json"), "r") as f:
            captions = json.load(f)["annotations"]
        image_id_to_objects: dict[int, set[str]] = {
            c["image_id"]: set() for c in captions if c["image_id"] in self.image_ids
        }
        for c in tqdm(captions, total=len(captions)):
            if c["image_id"] in self.image_ids:
                image_id_to_objects[c["image_id"]].update(self.caption_to_objects(c["caption"])[1])
        return image_id_to_objects

    def load_mappings(self):
        "load total objects and two mappings: synonym_map and double_word_map"
        with open(self.synonyms_path, "r") as f:
            synonyms = f.readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.total_coco_objects = set()  # mscoco objects and *all* synonyms
        self.synonym_map = {}
        for synonym in synonyms:
            for s in synonym:
                s = s.strip()
                self.total_coco_objects.add(s)
                self.synonym_map[s] = synonym[0]

        coco_double_words = [word for word in self.synonym_map.keys() if len(word.strip().split(" ")) >= 2]
        coco_double_words += ["home plate", "train track"]

        animal_words = [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]
        vehicle_words = ["jet", "train"]

        self.double_word_map = {}
        for double_word in coco_double_words:
            self.double_word_map[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_map["baby %s" % animal_word] = animal_word
            self.double_word_map["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_map["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_map["bow tie"] = "tie"
        self.double_word_map["toilet seat"] = "toilet"
        self.double_word_map["wine glas"] = "wine glass"

    def caption_to_objects(self, caption: str):
        """
        Input: caption
        Output: MSCOCO words in the caption
        """
        words = [self.lemma.lemmatize(w) for w in nltk.word_tokenize(caption.lower())]

        # replace double words
        i = 0
        double_words = []
        # idxs = []
        while i < len(words):
            # idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_map:
                double_words.append(self.double_word_map[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # get synonyms for all words in the caption
        # idxs = [idxs[idx] for idx, word in enumerate(words) if word in self.total_objects]
        words = [word for word in words if word in self.total_coco_objects]
        word_category = [self.synonym_map[word] for word in words]
        return words, word_category

    def compute(self, image_ids: list[int], captions: list[str]):
        "compute chair_s and chair_i from generated captions"
        assert len(image_ids) == len(captions)
        num_total_obj = 0
        num_hal_obj = 0
        num_hal_sent = 0
        num_caption_words = 0
        num_caption_chars = 0
        for image_id, caption in zip(image_ids, captions):
            num_caption_words += len(caption.split())
            num_caption_chars += len(caption)
            obj_set = set()
            num_sample_hal_obj = 0
            gold_category_set = self.gold_objects[image_id]
            for obj, category in zip(*self.caption_to_objects(caption)):
                if obj in obj_set:
                    continue  # repeated object counts only once
                obj_set.add(obj)
                num_sample_hal_obj += int(category not in gold_category_set)
            num_total_obj += len(obj_set)
            num_hal_obj += num_sample_hal_obj
            num_hal_sent += int(num_sample_hal_obj > 0)
        return num_total_obj, num_hal_obj, num_hal_sent, num_caption_words, num_caption_chars


def main():
    print("loading eval...")
    with open(args.caption_eval_path, "r") as f:
        content: list[str] = f.read().splitlines()
    image_ids, captions = [], []
    eval_end_pos = args.default_eval_samples if args.end_pos == int(1e10) else args.end_pos
    for line in content[args.start_pos : eval_end_pos]:
        try:
            image_name, caption = line.replace("### gpt: ", "").split("###")[:2]
        except ValueError as e:
            print(f"Skipping line {line} due to {e}")
            continue
        image_ids.append(int(image_name.split("_")[-1].split(".")[0]))
        captions.append(caption)

    print("loading ground truth...")
    with open(os.path.join(args.annotation_path, "captions_train2014.json"), "r") as f:
        data = json.load(f)
    id2caption = {}
    for image in data["annotations"]:
        id2caption[image["image_id"]] = image["caption"]
    ground_truth = [id2caption[image_id] for image_id in image_ids]

    print("computing chair...")
    chair = CHAIR(args.annotation_path, args.synonyms_path, image_ids)
    total_obj, hal_obj, hal_sent, n_word, n_char = chair.compute(image_ids, captions)

    print("computing bleu... ")
    bleu_1, bleu_2 = bleu(captions, ground_truth)

    print("computing distinct... ")
    distinct1, distinct2 = distinct(captions)

    print(
        f"Total objects: {total_obj}"
        f"\nHallucinated objects: {hal_obj}"
        f"\nTotal sentences: {len(image_ids)}"
        f"\nHallucinated sentences: {hal_sent}"
        f"\nCHAIRs: {hal_sent / len(image_ids)}"
        f"\nCHAIRi: {hal_obj / total_obj}"
        f"\nAverage caption word len: {n_word / len(image_ids)}"
        f"\nAverage caption char len: {n_char / len(image_ids)}"
        f"\nBleu_1 Score: {bleu_1}"
        f"\nBleu_2 Score: {bleu_2}"
        f"\nDistinct-1: {distinct1}"
        f"\nDistinct-2: {distinct2}"
    )


if __name__ == "__main__":
    main()
