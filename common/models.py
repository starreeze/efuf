# -*- coding: utf-8 -*-
# @Date    : 2024-01-06 18:38:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"specific model behavior (model loading, data processing, etc.)"

from __future__ import annotations
from itertools import product
import os, re, torch, transformers
from functools import partial
from torch.nn import CrossEntropyLoss
from common.args import args
from dataclasses import dataclass, field
from typing import Optional


def load_ckpt(model, ckpt, device="cuda"):
    "load the trainable part from ckpt"
    if ckpt == getattr(args, f"{args.model}_path"):
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
    kwargs = {"length_penalty": args.generate_length_penalty} if args.generate_length_penalty != -1 else {}
    return model.generate(
        images,
        texts,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.generate_num_beams,
        temperature=args.generate_temperature,
        **kwargs,
    )


def blip_generate(model, texts, images):
    results = []
    kwargs = {"length_penalty": args.generate_length_penalty} if args.generate_length_penalty != -1 else {}
    for text, image in zip(texts, images):
        res = model.generate(
            {"prompt": [text], "image": image.unsqueeze(0)},
            max_length=args.max_new_tokens,
            temperature=args.generate_temperature,
            use_nucleus_sampling=True,
            num_beams=args.generate_num_beams,
            **kwargs,
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


class OWl:
    def load(
        self, ckpt: str, device="cuda", train=False, model_args: list[str] = []
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        from Owl.pipeline.interface import get_model
        from Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration

        model: MplugOwlForConditionalGeneration
        model, tokenizer, processor = get_model(args.owl_path, use_bf16=(args.train_dtype_str == "bfloat16"))  # type: ignore
        load_ckpt(model, ckpt, device)
        model.to(device)  # type: ignore
        if train:
            model.train().to(args.train_dtype)
            model.language_model = model.language_model.eval()
            model.vision_model = model.vision_model.eval()
            for param in model.language_model.parameters():
                param.requires_grad = False
            for param in model.vision_model.parameters():
                param.requires_grad = False
        else:
            model.eval()
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        return model, lambda image: processor(text=None, images=[image], return_tensors="pt")["pixel_values"][0]  # type: ignore

    def preprocess_text(self, input: str, output: str, add_end_sym) -> dict:
        "参考了Owl/pipeline/data_utils/xgpt_data_utils.py/MutiModalDataset中_extract_text_token_from_conversation的实现"

        media_tokens = ["<image>"]
        self.media_tokens = {k: -int(i + 1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {"<image>": 1 + 64}
        self.max_length = 512
        enc_chunk = []  # tokenize后的input_ids（包含了input文本和output文本）
        enc_length = 0  # tokenize后的input_ids的长度
        prompt_length = -2
        if self.tokenizer.bos_token_id > 0:  # type: ignore
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        pattern = "|".join(map(re.escape, list(self.media_tokens.keys()) + ["AI: ", "\nHuman: "]))
        input_chunk_strs = re.split(f"({pattern})", input)
        input_chunk_strs = [x for x in input_chunk_strs if len(x) > 0]
        label_chunk = []

        for idx, chunk_str in enumerate(input_chunk_strs):
            if idx == 0:
                enc_chunk = prompt_chunk + self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]  # type: ignore
                enc_length = len(enc_chunk)
                label_chunk = [
                    0
                ] * enc_length  # input文本对应的input_ids部分的标签为0，为了之后生成mask，便于在loss计算中不参与计算
            else:
                if chunk_str in self.media_tokens:
                    enc_chunk += [self.media_tokens[chunk_str]] * self.media_lengths[chunk_str]
                    enc_length += self.media_lengths[chunk_str]
                    label_chunk += [0] * self.media_lengths[chunk_str]  # type: ignore
                else:
                    if input_chunk_strs[idx - 1] == "AI:":
                        curr_chunk: list[int] = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]  # type: ignore
                        # if True:
                        #     if enc_length + len(curr_chunk) >= self.max_length:
                        #         curr_chunk = curr_chunk[:self.max_length-enc_length]
                        #         curr_chunk += [self.tokenizer.eos_token_id]#!: 删去了这个位置的添加
                        # else:
                        if enc_length + len(curr_chunk) >= self.max_length + 1:
                            curr_chunk = curr_chunk[: self.max_length + 1 - enc_length]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]  # type: ignore
                        if enc_length + len(curr_chunk) >= self.max_length + 1:
                            curr_chunk = curr_chunk[: self.max_length + 1 - enc_length]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
        #! 改了一下self.tokenizer.eos_token_id的添加位置
        output_chunk: list[int] = self.tokenizer(output, add_special_tokens=False)["input_ids"]  # type: ignore
        if add_end_sym:
            if enc_length + len(output_chunk) >= self.max_length:
                output_chunk = output_chunk[: self.max_length - enc_length]
            output_chunk += [self.tokenizer.eos_token_id]  # type: ignore
        else:
            if enc_length + len(output_chunk) >= self.max_length + 1:
                output_chunk = output_chunk[: self.max_length + 1 - enc_length]
        enc_length += len(output_chunk)
        enc_chunk += output_chunk
        label_chunk += [1] * len(output_chunk)

        # 将总长度pad到self.max_length(也就是512)
        if enc_length < self.max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * (self.max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (self.max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == self.max_length + 1
        assert len(label_chunk) == self.max_length + 1

        # 计算loss 时用到的mask
        non_padding_mask = [1 if i < enc_length - 1 else 0 for i in range(self.max_length)]  # mask掉padding部分

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk[1:]).long()  # mask掉input文本部分
        prompt_length = torch.tensor([prompt_length]).long()

        tmp_enc_chunk = enc_chunk.clone()
        tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
        tmp_enc_chunk[tmp_enc_chunk < 0] = 0
        non_media_mask = torch.tensor(
            tmp_enc_chunk
        ).long()  # mask掉'<image>'部分(好像没必要，因为在ouput文本中不会出现'<image>'应该)
        non_media_mask = non_media_mask[1:].long()

        # model在forward中最终计算loss时候用的mask是 non_padding_mask * non_media_mask * prompt_mask

        return {
            "input_ids": enc_chunk,
            "prompt_length": prompt_length,
            "seq_length": enc_length,
            "non_padding_mask": non_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
        }

    def data_map(self, inputs: dict, add_end_sym=True) -> dict:
        global debug_print_for_1
        final_text_input = self.preprocess_text(inputs["input"], inputs["output"], add_end_sym=add_end_sym)
        return {
            "image": inputs["image"].unsqueeze(0),
            "text": final_text_input,
            "score": inputs["score"],
        }

    def collator(self, batch: list[dict]) -> dict:
        "参考了Owl/pipeline/utils.py中batchfy的实现"
        image = [data["image"] if data["image"] is not None else None for data in batch]
        if all([img is None for img in image]):
            image = None
        else:
            image = torch.cat([img for img in image if img is not None], dim=0)
        num_images_per_sample = torch.LongTensor(
            [data["image"].size(0) if data["image"] is not None else 0 for data in batch]
        )

        text = torch.stack([torch.LongTensor(data["text"]["input_ids"]) for data in batch], dim=0)
        non_padding_mask = torch.stack([torch.LongTensor(data["text"]["non_padding_mask"]) for data in batch], dim=0)
        non_media_mask = torch.stack([torch.LongTensor(data["text"]["non_media_mask"]) for data in batch], dim=0)
        prompt_mask = torch.stack([torch.LongTensor(data["text"]["prompt_mask"]) for data in batch], dim=0)

        output_batch = {
            "pixel_values": image,
            "image": image,
            "input_ids": text.long(),
            "labels": text.long().clone(),
            "num_images": num_images_per_sample.long(),
            "non_padding_mask": non_padding_mask.long(),
            "non_media_mask": non_media_mask.long(),
            "prompt_mask": prompt_mask.long(),
            "attention_mask": text.ne(self.tokenizer.pad_token_id).to(torch.long),  # type: ignore
            "score": torch.tensor([x["score"] for x in batch]).unsqueeze(1),
        }
        return output_batch

    @staticmethod
    def forward(model: torch.nn.Module, samples: dict, add_end_sym=False):
        result, labels, loss_mask = model(
            pixel_values=samples["pixel_values"],
            input_ids=samples["input_ids"],
            labels=samples["labels"],
            num_images=samples["num_images"],
            non_padding_mask=samples["non_padding_mask"],
            non_media_mask=samples["non_media_mask"],
            prompt_mask=samples["prompt_mask"],
            attention_mask=samples["attention_mask"],
            return_dict=True,
        )  # 更改了一下原模型中的forward的逻辑，增添了labels和loss_mask的返回

        logits = result["logits"]  # 8,2049, 32000
        bs = logits.shape[0]  # 8
        shift_logits = logits[..., :-1, :].contiguous()  # 8, 2048, 32000
        shift_labels = labels[..., 1:].contiguous()  # 8,2048
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, model.language_model.config.vocab_size)  # type: ignore
        shift_labels = shift_labels.view(-1)  # 8*2048
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels).view(bs, -1)
        return loss

    def generate(self, model, texts: list[str], images: torch.Tensor):
        "参考了Owl/pipeline/interface.py中的do_generate函数，并在interface.py中进行了初步测试"
        answers = []
        for i in range(len(texts)):
            text = [texts[i]]
            image = images[i].unsqueeze(0)
            inputs = self.processor(text=text, images=None, return_tensors="pt")
            inputs["pixel_values"] = image
            inputs = {k: v.to(model.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            kwargs = {"length_penalty": args.generate_length_penalty} if args.generate_length_penalty != -1 else {}
            with torch.no_grad():
                res = model.generate(
                    **(inputs | kwargs),
                    max_length=512,
                    top_p=0.9,
                    temperature=args.generate_temperature,
                    num_beams=args.generate_num_beams,
                )
            sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            answers.append(sentence)
        return answers


class OWLlrv(OWl):
    def load(
        self, ckpt: str, device="cuda", train=False, model_args: list[str] = []
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        from Owl.pipeline.interface import get_model
        from Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration

        # from Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
        # from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore

        model: MplugOwlForConditionalGeneration
        model, tokenizer, processor = get_model(args.owllrv_path, use_bf16=(args.train_dtype_str == "bfloat16"))  # type: ignore
        print(f"loaded from {args.owllrv_path}")

        peft_config = LoraConfig(
            target_modules=r".*language_model.*\.(q_proj|v_proj)",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)  # type: ignore

        lora_path = args.owllrv_lora_path
        prefix_state_dict = torch.load(lora_path, map_location=device)

        unexpected = [
            f"base_model.model.language_model.model.layers.{i}.self_attn.{n}_proj.weight"
            for i, n in product(range(32), ("q", "v"))
        ]
        for k in unexpected:
            prefix_state_dict[k[: -len("weight")] + "base_layer.weight"] = prefix_state_dict[k]
            del prefix_state_dict[k]

        model.load_state_dict(prefix_state_dict)
        require_grad_para = []
        # print(f"loaded from lora_path, type={type(model)},\n\nmodel={model}")
        for name, p in model.named_parameters():
            if p.requires_grad:
                require_grad_para.append(name)

        # print(f"para_require={require_grad_para}, len={len(require_grad_para)}")

        load_ckpt(model, ckpt, device)
        print(f"load_ckpt from {args.owllrv_ckpt_load_path}")
        model.to(device)
        if train:
            model.requires_grad_(True)
            model.train().to(args.train_dtype)
            model.language_model = model.language_model.eval()
            model.vision_model = model.vision_model.eval()
            for param in model.language_model.parameters():
                param.requires_grad = False
            for param in model.vision_model.parameters():
                param.requires_grad = False
        else:
            model.eval()
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        return model, lambda image: processor(text=None, images=[image], return_tensors="pt")["pixel_values"][0]  # type: ignore


class LlavaModel:
    # No modification made to llava 1.5
    # shareGPT4V.forward is modified according to llava 1.5's version, to enable label output
    # Below is modified from the llava train script. ShareGPT4V adopts the same model.

    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default=getattr(args, f"{args.model}_path", None))
        version: Optional[str] = field(default="v1")
        freeze_backbone: bool = field(default=True)
        tune_mm_mlp_adapter: bool = field(default=True)
        vision_tower: Optional[str] = field(default=getattr(args, f"{args.model}_vit_path", None))
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

    def load(self, ckpt, device="cuda", train=False, model_args=[]):
        if args.model == "llava":
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as VLM
            from llava.mm_utils import tokenizer_image_token, process_images
            from llava.constants import IGNORE_INDEX
        elif args.model == "share4v":
            from share4v.model.language_model.share4v_llama import Share4VLlamaForCausalLM as VLM
            from share4v.mm_utils import tokenizer_image_token, process_images
            from share4v.constants import IGNORE_INDEX
        else:
            raise NotImplementedError()

        parser = transformers.HfArgumentParser((self.ModelArguments, self.DataArguments, self.TrainingArguments))  # type: ignore
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            ["--output_dir", getattr(args, f"{args.model}_ckpt_save_path")]
        )

        dtype = args.train_dtype if train else torch.float16
        model: VLM = VLM.from_pretrained(  # XXX error!
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
            module.eval()
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
        if args.model == "share4v":  # no position_ids for share4v
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = embed_image(
                samples["input_ids"], samples["attention_mask"], None, samples["labels"], samples["image"]
            )
        else:
            input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = embed_image(
                samples["input_ids"], None, samples["attention_mask"], None, samples["labels"], samples["image"]
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
        # input_ids: torch.Tensor = self.tokenize_image(texts[0])  # type: ignore
        # as the prompt are all the same, it can be copied from the first prompt
        # mind the padding if want to modify it into vqa
        # input_ids = input_ids.unsqueeze(0).expand([len(texts), -1]).cuda()
        results = []
        kwargs = {"length_penalty": args.generate_length_penalty} if args.generate_length_penalty != -1 else {}
        for text, image in zip(texts, images):
            input_ids: torch.Tensor = self.tokenize_image(text).cuda().unsqueeze(0)  # type: ignore
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    temperature=args.generate_temperature,
                    top_p=0.9,
                    num_beams=args.generate_num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=args.max_new_tokens,
                    **kwargs,
                )
            result = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
            # as the prompt is in the format of ### human/gpt, we need to truncate the generation to the next ###
            # results.append(result.replace("### gpt: ", "").split("###")[0])
            results.append(result)
        return results

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


class LlavaHaDpo(LlavaModel):
    def load(self, ckpt, device="cuda", train=False, model_args=[]):
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as VLM
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.constants import IGNORE_INDEX
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

        lora_cfg_pretrained = AutoConfig.from_pretrained(args.llava_lora_path)

        parser = transformers.HfArgumentParser((self.ModelArguments, self.DataArguments, self.TrainingArguments))  # type: ignore
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            ["--output_dir", getattr(args, f"{args.model}_ckpt_save_path")]
        )

        dtype = args.train_dtype if train else torch.float16
        # model: VLM = VLM.from_pretrained(  # XXX error!
        #     args.llava_path, local_files_only=True, device_map={"": device}, torch_dtype=dtype
        # )  # type: ignore
        model = VLM.from_pretrained(
            args.llava_path,
            config=lora_cfg_pretrained,
            # low_cpu_mem_usage=True,
            local_files_only=True,
            device_map={"": device},
            torch_dtype=dtype,
        )
        model.config.use_cache = False
        model.model.requires_grad_(False)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            getattr(args, f"{args.model}_path"),
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        non_lora_trainables = torch.load(
            os.path.join(args.llava_lora_path, "non_lora_trainables.bin"), map_location="cpu"
        )
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
        }
        non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        try:
            model.load_state_dict(non_lora_trainables, strict=False)
        except BaseException:
            pass
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.llava_lora_path)
        model = model.merge_and_unload()

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
        # load_ckpt(model, ckpt, device)
        model.train(train)

        freeze_modules: list[torch.nn.Module] = [
            model.model.vision_tower,  # type: ignore
            model.model.embed_tokens,
            model.model.norm,
            model.model.layers,
            model.lm_head,
        ]
        for module in freeze_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        return model, lambda x: vis_processor.preprocess(x, return_tensors="pt")["pixel_values"][0]


class Llavarlhf(LlavaModel):
    def load(self, ckpt, device="cuda", train=False, llava_args=[]):
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as VLM
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IGNORE_INDEX

        parser = transformers.HfArgumentParser((self.ModelArguments, self.DataArguments, self.TrainingArguments))  # type: ignore
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            ["--output_dir", getattr(args, f"{args.model}_ckpt_save_path")]
        )

        dtype = args.train_dtype if train else torch.float16
        model: VLM = VLM.from_pretrained(
            args.llavarlhf_path, local_files_only=True, device_map={"": device}, torch_dtype=dtype
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
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        # if not train:
        #     model.requires_grad_(False)
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(f"name={name}, p.requires_grad={p.requires_grad}")
        return model, lambda x: vis_processor.preprocess(x, return_tensors="pt")["pixel_values"][0]


llava_model = LlavaModel()
llavarlhf_model = Llavarlhf()
llavahadpo_model = LlavaHaDpo()
owl_model = OWl()
owllrv_model = OWLlrv()
model_loaders = {
    "minigpt": load_minigpt,
    "blip": load_blip,
    "llava": llava_model.load,
    "share4v": llava_model.load,
    "owl": owl_model.load,
    "owllrv": owllrv_model.load,
    "llavarlhf": llavarlhf_model.load,
    "llavahadpo": llavahadpo_model.load,
}
data_maps = {
    "minigpt": minigpt_data_map,
    "blip": blip_data_map,
    "llava": llava_model.data_map,
    "share4v": llava_model.data_map,
    "owl": owl_model.data_map,
    "owllrv": owllrv_model.data_map,
    "llavarlhf": llavarlhf_model.data_map,
    "llavahadpo": llavahadpo_model.data_map,
}
sample_collators = {
    "minigpt": None,
    "blip": None,
    "llava": llava_model.collator,
    "share4v": llava_model.collator,
    "owl": owl_model.collator,
    "owllrv": owllrv_model.collator,
    "llavarlhf": llavarlhf_model.collator,
    "llavahadpo": llavahadpo_model.collator,
}
batch_collators = {
    "minigpt": merge_batch_collator,
    "blip": merge_batch_collator,
    "llava": llava_model.pad_batch_collator,
    "share4v": llava_model.pad_batch_collator,
    "owl": merge_batch_collator,
    "owllrv": merge_batch_collator,
    "llavarlhf": llavarlhf_model.pad_batch_collator,
    "llavahadpo": llavahadpo_model.pad_batch_collator,
}
generators = {
    "minigpt": minigpt_generate,
    "blip": blip_generate,
    "llava": llava_model.generate,
    "share4v": llava_model.generate,
    "owl": owl_model.generate,
    "owllrv": owllrv_model.generate,
    "llavarlhf": llavarlhf_model.generate,
    "llavahadpo": llavahadpo_model.generate,
}
model_forward = {
    "minigpt": lambda model, samples, add_end_sym: model(samples, add_end_sym=add_end_sym, reduction="none")["loss"],
    "blip": lambda model, samples, add_end_sym: model(samples, add_end_sym=add_end_sym, reduction="none")["loss"],
    "llava": llava_model.forward,
    "share4v": llava_model.forward,
    "owl": owl_model.forward,
    "owllrv": owllrv_model.forward,
    "llavarlhf": llavarlhf_model.forward,
    "llavahadpo": llavahadpo_model.forward,
}


def main():
    pass


if __name__ == "__main__":
    main()
