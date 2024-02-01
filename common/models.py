# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 18:38:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"specific model behavior (model loading, data processing, etc.)"

from __future__ import annotations
import os, torch, transformers
from functools import partial
from torch.nn import CrossEntropyLoss
from common.args import args


def load_ckpt(model, ckpt, device="cuda"):
    "load the trainable part from ckpt"
    original_path_name = "llava_path" if args.model == "llava" else f"{args.model}_ckpt_load_path"
    if ckpt == getattr(args, original_path_name):
        print(f"Already loaded the original version from {ckpt}")
        return
    latest = ckpt if os.path.isfile(ckpt) else os.path.join(ckpt, sorted(os.listdir(ckpt))[-1])
    print(f"Loading from {latest}")
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)


def load_minigpt(
    ckpt, device="cuda", train=False, model_args: list[str] = []
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from common.args import minigpt4_finetune_parser
    from minigpt4.common.eval_utils import init_model

    model, vis_processor = init_model(minigpt4_finetune_parser().parse_args(model_args), device)
    load_ckpt(model, ckpt, device)
    model = model.train().to(args.train_dtype) if train else model
    return model, vis_processor


def load_blip(ckpt, device="cuda", train=False, model_args: list[str] = []):
    from lavis.models import load_model_and_preprocess
    from lavis.common.registry import registry
    from lavis.models.blip2_models.blip2 import disabled_train

    registry.register_path("library_root", "lavis")
    model, vis_processor, _ = load_model_and_preprocess(
        "blip2_vicuna_instruct", "vicuna7b", is_eval=not train, device=device
    )
    load_ckpt(model, ckpt, device)
    if train:
        model.train().to(args.train_dtype)
        model.llm_model = model.llm_model.eval()
        for param in model.llm_model.parameters():
            param.requires_grad = False
        model.llm_model.train = disabled_train
    else:
        model.eval()
    return model, vis_processor["train"]  # type: ignore


def load_llava(ckpt, device="cuda", train=False, model_args: list[str] = []):
    return llava_model.load(ckpt, device, train)


def minigpt_data_map(inputs: dict, add_end_sym=None) -> dict:
    return {
        "image": inputs["image"],
        "instruction_input": inputs["input"],
        "answer": inputs["output"],
        "score": inputs["score"],
    }


def blip_data_map(inputs: dict, add_end_sym=None) -> dict:
    return {
        "image": inputs["image"],
        "text_input": inputs["input"],
        "text_output": inputs["output"],
        "score": inputs["score"],
    }


def minigpt_generate(model, texts, images):
    return model.generate(images, texts, max_new_tokens=args.max_new_tokens)


def blip_generate(model, texts, images):
    results = []
    for text, image in zip(texts, images):
        res = model.generate(
            {"prompt": [text], "image": image.unsqueeze(0)},
            max_length=args.max_new_tokens,
            temperature=0.9,
            use_nucleus_sampling=True,
        )
        results.extend(res)
    return results


def merge_batch_collator(*inputs: dict):
    "a higher level collator that collate multiple batches into one, by directly merging the inputs"
    ret = {}
    for k in inputs[0].keys():
        if isinstance(inputs[0][k], torch.Tensor):
            ret[k] = torch.cat([x[k] for x in inputs])
        elif isinstance(inputs[0][k], list):
            ret[k] = sum([x[k] for x in inputs], [])
        else:
            raise NotImplementedError(f"Unknown type when concatenating: {type(inputs[0][k]).__name__}")
    return ret


class LlavaModel:
    # No modification made to llava 1.5
    # shareGPT4V.forward is modified according to llava 1.5's version, to enable label output

    def load(self, ckpt, device="cuda", train=False, llava_args=[]):
        if args.model == "llava":
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as VLM
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IGNORE_INDEX
        elif args.model == "share4v":
            from share4v.model.language_model.share4v_llama import Share4VLlamaForCausalLM as VLM
            from share4v.mm_utils import tokenizer_image_token
            from share4v.constants import IGNORE_INDEX
        else:
            raise NotImplementedError()

        # Below is modified from the llava train script. ShareGPT4V adopts the same model.
        from dataclasses import dataclass, field
        from typing import Optional

        @dataclass
        class ModelArguments:
            model_name_or_path: Optional[str] = field(default=getattr(args, f"{args.model}_path"))
            version: Optional[str] = field(default="v1")
            freeze_backbone: bool = field(default=True)
            tune_mm_mlp_adapter: bool = field(default=True)
            vision_tower: Optional[str] = field(default=getattr(args, f"{args.model}_vit_path"))
            mm_vision_select_layer: Optional[int] = field(default=-2)  # default to the last layer
            pretrain_mm_mlp_adapter: Optional[str] = (
                field(default=os.path.join(args.llava_path, "mm_projector.bin")) if args.model == "llava" else None
            )
            mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
            mm_use_im_start_end: bool = field(default=False)
            mm_use_im_patch_token: bool = field(default=False)
            mm_vision_select_feature: Optional[str] = field(default="patch")

        @dataclass
        class DataArguments:
            lazy_preprocess: bool = True
            is_multimodal: bool = True
            image_aspect_ratio: str = "pad"

        @dataclass
        class TrainingArguments(transformers.TrainingArguments):
            optim: str = field(default="adamw_torch")
            remove_unused_columns: bool = field(default=False)
            freeze_mm_mlp_adapter: bool = field(default=False)
            mpt_attn_impl: Optional[str] = field(default="triton")
            model_max_length: int = field(
                default=4096,
                metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
            )
            double_quant: bool = field(
                default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
            )
            quant_type: str = field(
                default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
            )
            bits: int = field(default=16, metadata={"help": "How many bits to use."})
            lora_enable: bool = False
            lora_r: int = 64
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_weight_path: str = ""
            lora_bias: str = "none"
            mm_projector_lr: Optional[float] = None
            group_by_modality_length: bool = field(default=True)

        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            ["--output_dir", getattr(args, f"{args.model}_ckpt_save_path")]
        )

        dtype = args.train_dtype if train else torch.float16
        model: VLM = VLM.from_pretrained(
            args.llava_path, local_files_only=True, device_map={"": device}, torch_dtype=dtype
        )  # type: ignore
        model.config.use_cache = False
        model.model.requires_grad_(False)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            getattr(args, f"{args.model}_path"),
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token

        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=dtype, device=device)

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = True
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        self.ignore_value = IGNORE_INDEX
        self.tokenizer = tokenizer
        vis_processor = model.get_model().get_vision_tower().image_processor  # type: ignore
        self.tokenize_image = partial(tokenizer_image_token, tokenizer=self.tokenizer, return_tensors="pt")
        load_ckpt(model, ckpt, device)
        model.train(train)

        freeze_modules: list[torch.nn.Module] = [
            model.model.vision_tower,  # type: ignore
            model.model.embed_tokens,
            model.model.norm,
            model.model.layers,
            model.lm_head,
        ]
        for module in freeze_modules:
            for param in module.parameters():
                param.requires_grad = False

        return model, lambda x: vis_processor.preprocess(x, return_tensors="pt")["pixel_values"][0]

    def data_map(self, inputs: dict, add_end_sym=True) -> dict:
        input_ids = self.tokenize_image(inputs["input"])
        output_ids = self.tokenizer(inputs["output"], add_special_tokens=False, return_tensors="pt").input_ids[0]
        if add_end_sym:
            output_ids = torch.cat(
                [output_ids, self.tokenizer.eos_token_id * torch.ones(len(output_ids), dtype=torch.long)]
            )
        return {
            "image": inputs["image"],
            "input_ids": torch.cat([input_ids, output_ids]),  # type: ignore
            "labels": torch.cat([torch.ones(len(input_ids), dtype=torch.long) * self.ignore_value, output_ids]),
            "score": inputs["score"],
        }

    def collator(self, batch: list[dict]) -> dict:
        "the ordinary collator to construct a batch"
        input_ids, labels = tuple([instance[key] for instance in batch] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_value)
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).to(torch.long),  # type: ignore
            "score": torch.tensor([x["score"] for x in batch]),
        }

    def forward(self, model, samples, _):
        embed_image = model.prepare_inputs_labels_for_multimodal
        if args.model == "llava":
            input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = embed_image(
                samples["input_ids"], None, samples["attention_mask"], None, samples["labels"], samples["image"]
            )
        else:  # no position_ids for share4v
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = embed_image(
                samples["input_ids"], samples["attention_mask"], None, samples["labels"], samples["image"]
            )
        del samples["input_ids"], samples["attention_mask"], samples["labels"], samples["image"]
        logits: torch.Tensor = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )["logits"]
        bs = labels.shape[0]

        # The below code is modified from transformers.models.llama.modeling_llama.LlamaForCasualLM.forward
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels).view(bs, -1)  # [bs, seq_len]

        # bypass the input and padding token
        # since minigpt and blip take str as inputs, we only need this here
        masks = shift_labels.ne(self.ignore_value).to(torch.float).view(bs, -1)
        loss = torch.sum(loss * masks, dim=1) / torch.clamp(masks.sum(dim=1), min=1e-6)
        return loss

    def generate(self, model, texts, images):
        input_ids: torch.Tensor = self.tokenize_image(texts[0])  # type: ignore
        # as the prompt are all the same, it can be copied from the first prompt
        # mind the padding if want to modify it into vqa
        input_ids = input_ids.unsqueeze(0).expand([len(texts), -1]).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images.half().cuda(),
                temperature=1,
                top_p=0.9,
                num_beams=5,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
            )
        texts = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)
        # as the prompt is in the format of ### human/gpt, we need to truncate the generation to the next ###
        return [text.replace("### gpt: ", "").split("###")[0] for text in texts]

    def pad_batch_collator(self, *inputs: dict):
        "a higher level collator that collate multiple batches into one, by padding the ids/masks and merging other data"
        pad_value = {"input_ids": self.tokenizer.pad_token_id, "attention_mask": 0, "labels": self.ignore_value}
        ret = {}
        for k in inputs[0].keys():
            assert isinstance(inputs[0][k], torch.Tensor)
            if torch.is_floating_point(inputs[0][k]):  # images, scores
                ret[k] = torch.cat([x[k] for x in inputs])
            else:
                samples = [sample for x in inputs for sample in x[k]]
                ret[k] = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=pad_value[k])
        return ret


llava_model = LlavaModel()

model_loaders = {
    "minigpt": load_minigpt,
    "blip": load_blip,
    "llava": llava_model.load,
    "share4v": llava_model.load,
}
data_maps = {
    "minigpt": minigpt_data_map,
    "blip": blip_data_map,
    "llava": llava_model.data_map,
    "share4v": llava_model.data_map,
}
sample_collators = {
    "minigpt": None,
    "blip": None,
    "llava": llava_model.collator,
    "share4v": llava_model.collator,
}
batch_collators = {
    "minigpt": merge_batch_collator,
    "blip": merge_batch_collator,
    "llava": llava_model.pad_batch_collator,
    "share4v": llava_model.pad_batch_collator,
}
generators = {
    "minigpt": minigpt_generate,
    "blip": blip_generate,
    "llava": llava_model.generate,
    "share4v": llava_model.generate,
}
model_forward = {
    "minigpt": lambda model, samples, add_end_sym: model(samples, add_end_sym=add_end_sym, reduction="none")["loss"],
    "blip": lambda model, samples, add_end_sym: model(samples, add_end_sym=add_end_sym, reduction="none")["loss"],
    "llava": llava_model.forward,
    "share4v": llava_model.forward,
}


def main():
    pass


if __name__ == "__main__":
    main()
