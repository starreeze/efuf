from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm
import json
import openai, g4f

openai.api_key = "sk-47lztbvTmb43q1lVanVebwcFtG9R3nGYb1eeZUFXaci4XEe0"
openai.api_base = "https://api.chatanywhere.com.cn/v1"
proxy = "http://127.0.0.1:7890"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from common.utils import merge_dict_set
import argparse
import json
import time


template = '''your response is just only the score, no need for explanation, Evaluate whether a given caption accurately represents the main content and objects of 
an associated image with scale 1-100
### Image Url
{}

### Captions
{}
'''

class Chat:
    def __init__(self, model="", timeout_sec=20, openai_apikey=''):
        self.model = model
        self.timeout = timeout_sec
        openai.api_key = openai_apikey

    def chat_completion(self, messages, temperature=0.2, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0):

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )

        return response
        
if __name__ == '__main__':
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
    parser = argparse.ArgumentParser()
    with open(os.path.join(args.annotation_path, "captions_train2014.json"), "r") as f:
        data=json.load(f)
    ground_truth = []

    image_url = []
    for image_id in image_ids:
        for image in data["images"]:
            if image["id"] == image_id:
                image_url.append(image["coco_url"])
                break

    chat = Chat(model=args.gpt_model, timeout_sec=100, openai_apikey=openai.api_key)

    # ask GPT-4 to evaluate
    scores = []
    for i in range(len(image_ids)):
        input_text = template.format(image_url[i], captions[i])
        response = chat.chat_completion(
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.0,
                )
        print(f'Response: {response}')
        scores.append(response)        

    # # assuming order of 96 questions is not changed
    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))

    