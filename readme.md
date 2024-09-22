This repo is the official code for paper **EFUF: Efficient Fine-grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models**.

**news**

- 2024.09.20: Our paper is accepted by EMNLP 2024 (main conference)!
- 2024.09.23: We finished releasing the code and the datasets.
  <!-- - 2024.09.24: Checkpoints are released for better reproductivity. -->

## Overview

![](assets/method.png)

## Checkpoints

We are planning to release the checkpoints of our models. Stay tuned!

## Install

First, clone this repository and navigate to efuf folder.

```bash
git clone https://github.com/starreeze/efuf.git
cd efuf
```

### Docker

We recommend to use docker to prepare the environment.

1. Build the docker image

```bash
cd deploy
docker build -t efuf:1.0 .
```

If your machine cannot connect to github to download the flash attention pip wheel, you can download it manually on https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl and put it to `deploy/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`.

2. To start the container, run the following command in the project root directory

```bash
docker run --gpus all --ipc=host --network=host --rm -it -v .:/workspace efuf:1.0
```

More `-v` options can be added to mount the data and output directories.

### Conda

```Shell
conda create -n efuf python=3.10 -y
conda activate efuf
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r deploy/requirements.txt
```

Finally, you need to install flash-attention manually before running the model.

## Datasets Preparation

To reproduce our results, you need to prepare the datasets first.

1. Download LLaVA stage 2 training data, and put it to `dataset/llava`. The dir tree should contain at least the following:

   ```
   dataset/llava
   ├── coco
   │   └── train2017 [118287 entries exceeds filelimit, not opening dir]
   ├── gqa
   │   └── images [148854 entries exceeds filelimit, not opening dir]
   ├── llava_v1_5_mix665k.json
   ├── ocr_vqa
   │   └── images [207927 entries exceeds filelimit, not opening dir]
   ├── textvqa
   │   └── train_images [25119 entries exceeds filelimit, not opening dir]
   └── vg
       ├── VG_100K [64346 entries exceeds filelimit, not opening dir]
       └── VG_100K_2 [43903 entries exceeds filelimit, not opening dir]
   ```

2. Download the MSCOCO 2014 train images and annotations, and put it to `dataset/images` and `dataset/annotations`. The dir tree should contain at least the following:

   ```
   dataset
   ├── llava [...]
   ├── images [82783 entries exceeds filelimit, not opening dir]
   └── annotations
       ├── captions_train2014.json
       ├── captions_val2014.json
       ├── instances_train2014.json
       ├── instances_val2014.json
       ├── person_keypoints_train2014.json
       └── person_keypoints_val2014.json
   ```

3. Download our constructed dataset `pos_neg.json` `sentences.json` from [huggingface dataset](https://huggingface.co/datasets/starreeze/efuf-unlearning-30k) and put it to `dataset`. The dir tree should contain at least the following:
   ```
   dataset
   ├── llava [...]
   ├── images [...]
   ├── annotations [...]
   ├── pos_neg.json
   └── sentences.json
   ```

Note that `pos_neg.json` uses a special format, please refer to `finetune/format_data.py` for details.

```
from captions, objects and scores format a flattened json file,
where each object has an entry containing fields: sentence, sub-sentence/object position, score

normal text ... hallucination object/subsentence
                ^
                |
    sub-sentence/object position

sentence: type=str, stripped sentence until the target
sub-sentence/object mask: type=int, beginning position (char-level) of the unlearn target
score: float, clip score of the object
```

## Training

To start training, you need an A100-80G GPU.

```Shell
python finetune/train.py --model [model_name] --[model_name]_path [pretrained_weights] --wandb_user [username]
```

- model_name: `minigpt`, `llava`, `owl`, `share4v`.
- pretrained_weights: the path to the pretrained weights of the model, huggingface id or local path.
- wandb_user: your wandb username. If you want to see the train curve on wandb, you need to manually sync after training.

Example:

```Shell
python finetune/train.py --model llava --llava_path /path/to/llava
```

After training, the checkpoint will be saved in `checkpoints/model_name/run_name`. You can also look into `common/args.py` to see all the arguments.

## Evaluation

### Image Caption

First, run inference on COCO images. The first 1600 images that are not used in training will be used for evaluation.

```Shell
python evaluate/eval_caption.py --model [model_name] --[model_name]_path [pretrained_weights] --[model_name]_ckpt_load_path [checkpoint_path] --caption_eval_path [caption_path]
```

- checkpoint_path: the path to the checkpoint (obtained by running train in the last step). If this arg is omitted, it will evaluate on the original model instead.
- output_path: the path to save the output.

Then, run the evaluation script:

```Shell
python evaluate/eval_auto.py --caption_eval_path [caption_path]
```

### VQA

After obtaining the trained checkpoint, you can directly evaluate it on VQA and reasoning tasks.

```Shell
python evaluate/eval_[mme/gqa/sqa/qbench].py --model [model_name] --[model_name]_path [pretrained_weights] --[model_name]_ckpt_load_path [checkpoint_path] --run_name skip_train
```

The result will be printed in the terminal.

## Extensions

### Construct Your Own Unlearning Dataset

The dataset we provide is around 30k, constructed from responses of minigpt. If you want to construct your own unlearning dataset using more responses or from other models, you might want to look at the following code:

1. `insight/object.py`: extract object information from the image caption.
2. `insight/clip.py`: calculate image-relevance score for each object and response.
3. `finetune/format_data.py`: format the data to the unlearning format.

### Add Your Own Model

You can easily implement finetuning on other models using the EFUF framework. All you need to do is:

1. Implement loader, forward function and data maps for your model in `common/models.py`. Take the 4 models we officially support as examples.
2. Add the required arguments in `common/args.py`.

## Citation

If you find this helpful, please kindly consider citing our paper:

```
@misc{xing2024efuf,
title={EFUF: Efficient Fine-grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models},
author={Shangyu Xing and Fei Zhao and Zhen Wu and Tuo An and Weihao Chen and Chunhui Li and Jianbing Zhang and Xinyu Dai},
year={2024},
eprint={2402.09801},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```
