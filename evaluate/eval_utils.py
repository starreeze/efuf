from common.args import args


def get_eval_caption():
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
    return image_ids, captions
