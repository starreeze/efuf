# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 10:04:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""extract objects from the vqa/caption data"""

from __future__ import annotations
import json, re, os, sys, torch
from io import TextIOWrapper
from functools import partial
from abc import abstractmethod
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.interrupt_wrapper import resumable_fn
from common.args import args


class LlamaChat:
    model, tokenizer = None, None

    @classmethod
    def init_llama(cls):
        from transformers import LlamaForCausalLM, LlamaTokenizer

        cls.model = LlamaForCausalLM.from_pretrained(
            args.llama_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            load_in_8bit=args.llama_8bit,
            device_map={"": 0},
        )
        cls.tokenizer = LlamaTokenizer.from_pretrained(args.llama_path, use_fast=True, local_files_only=True)

    def __init__(self, instruction, max_new_tokens) -> None:
        self.prompt = args.llama_sys_prompt.replace(args.llama_instruction_placeholder, instruction)
        self.max_new_tokens = max_new_tokens

    def infer(self, inputs: str) -> str:
        inputs = self.prompt + inputs
        if self.model is None:
            self.init_llama()
        model_inputs = self.tokenizer(inputs, return_tensors="pt", return_token_type_ids=False).to(  # type:ignore
            "cuda:0"
        )
        output = self.model.generate(  # type:ignore
            **model_inputs,
            min_length=1,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            temperature=1,
            top_p=0.9,
            repetition_penalty=1,
            length_penalty=1,
            do_sample=False,
        )
        output_tokens = output[0][len(model_inputs["input_ids"][0]) :]
        response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)  # type:ignore
        return response


class LLMExtractor:
    def __init__(self, instruction_path) -> None:
        with open(instruction_path, "r") as f:
            self.instruction = f.read().format(
                input_label=args.prompt_input_label, output_label=args.prompt_output_label
            )

    @staticmethod
    def segment_objects(objects_str: str, caption: str) -> set[str]:
        if "i'm sorry" in objects_str.lower():
            raise ValueError("model not willing to respond.")

        # sometimes LLM will not follow the format and output "the objects in the image are: xxx, xxx..."
        # we should filter its description which is usually ends with :
        # if not found then [0 :] is also ok with this expression
        objects_str = objects_str[objects_str.find(":") + 1 :]
        if (p := objects_str.find("(")) != -1:  # we also discard everything after (
            objects_str = objects_str[:p]

        objects_str = re.sub("[\n\t\r]", " ", objects_str).strip(args.subsentence_splitter_set + "'\" ")
        objects_brackets = set()  # repetition is not allowed
        for obj in objects_str.split(args.object_splitter):
            try:  # object must be found as a whole word match
                match = re.search(r"\b" + obj + r"\b", caption)
                if match:
                    target = match.group(0)
                    if re.search(r"\[(\w )*" + obj + r"( \w)*\]", caption):
                        target = "[" + target + "]"
                    objects_brackets.add(target)
            except re.error:
                tqdm.write(f"matching failed for: {obj}")
        return objects_brackets

    @staticmethod
    def format_input(inputs: str) -> str:
        return f"{args.prompt_input_label} {inputs}\n{args.prompt_output_label} "

    @abstractmethod
    def extract(self, inputs: str) -> set[str]:
        pass


class LlamaExtractor(LLMExtractor):
    def __init__(self, instruction_path) -> None:
        super().__init__(instruction_path)
        self.chat = LlamaChat(self.instruction, 50)

    def extract(self, inputs: str) -> set[str]:
        response = self.chat.infer(self.format_input(inputs))
        return self.segment_objects(response, inputs)

    # @staticmethod
    # def post_processing():
    #     "to regulate the objects file"
    #     with open(args.object_data_path, "r") as f:
    #         lines = f.read().splitlines()
    #     new_content = []
    #     for i in range(len(lines)):
    #         if lines[i].startswith("COCO_train2014"):
    #             content = lines[i].split(args.column_splitter)[-1]
    #             if content:
    #                 new_content.append(lines[i].strip(","))
    #             elif i + 1 < len(lines) and lines[i + 1] and not lines[i + 1].startswith("COCO_train2014"):
    #                 new_content.append(lines[i] + lines[i + 1].strip(","))
    #     with open(args.object_data_path, "w") as f:
    #         # with open("objects.txt", "w") as f:
    #         f.write("\n".join(new_content))


class ChatGPTExtractor(LLMExtractor):
    def __init__(self, prompt_path, version=35) -> None:
        # from common.gpt_free import anychat_gpt_35, g4f_gpt_4

        # super().__init__(prompt_path)
        # if version == 4:
        #     self.complete = partial(g4f_gpt_4, stream=False)
        # elif version == 35:
        #     self.complete = anychat_gpt_35
        # else:
        #     raise NotImplementedError()

        from common.gpt import gpt_infer

        super().__init__(prompt_path)
        if version == 4:
            self.complete = partial(gpt_infer, model="gpt-4-turbo")
        elif version == 35:
            self.complete = partial(gpt_infer, model="gpt-3.5-turbo")
        else:
            raise NotImplementedError()

    def extract(self, inputs: str) -> set[str]:
        raise NotImplementedError("not modified into instruction")
        response = self.complete([{"role": "user", "content": self.instruction.format(inputs)}])  # type:ignore
        self.segment_objects(response)  # type:ignore
        return response  # type:ignore


def extract_sample_vqa(sample: dict, extractor: LLMExtractor, output_fd: TextIOWrapper):
    if sample["hallucination"]:
        norm_index = sample["preference"]
        hal_index = 3 - norm_index
        hal_obj = extractor.extract(sample[f"output_{hal_index}"])
        norm_obj = extractor.extract(sample[f"output_{norm_index}"])
        output_fd.write(args.column_splitter.join([sample["image"], hal_obj, norm_obj]) + "\n")
        output_fd.flush()


def get_caption_info(sample: str):
    image_name, caption = sample.split(args.column_splitter)
    rank = 0
    if caption[-1] == "*":
        caption = caption[:-1].strip()
        rank = 1
    elif caption[-1] == "#":
        caption = caption[:-1].strip()
        rank = -1
    return image_name, caption, rank


def extract_sample_caption(sample: str, extractor: LLMExtractor, output_fd: TextIOWrapper):
    # image name ### caption */#
    image_name, caption, rank = get_caption_info(sample)
    assert caption
    objects = extractor.extract(caption)
    objects_str = args.object_splitter.join(objects)
    output_fd.write(args.column_splitter.join([image_name, str(rank), objects_str]) + "\n")
    output_fd.flush()


def extract_vqa(extractor_type):
    with open(args.vqa_data_path, "r") as f:
        vqa_data = json.load(f)
    extractor = extractor_type(args.vqa_prompt_path)
    with open(args.object_data_path, "w" if args.restart else "a") as f:
        resumable_fn(partial(extract_sample_vqa, extractor=extractor, output_fd=f), vqa_data, restart=args.restart)


def extract_caption(extractor_type):
    with open(args.caption_data_path, "r") as f:
        captions = f.read().splitlines()
    extractor = extractor_type(args.object_extract_prompt_path)
    with open(args.object_data_path, "w" if args.restart else "a") as f:
        resumable_fn(partial(extract_sample_caption, extractor=extractor, output_fd=f), captions, restart=args.restart)


if __name__ == "__main__":
    extract_caption(LlamaExtractor)
    # LlamaExtractor.post_processing()
    # print(GPTExtractor().extract(open("tmp.txt", "r").read()))
