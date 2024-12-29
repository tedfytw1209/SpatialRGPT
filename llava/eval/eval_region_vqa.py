import argparse
import copy
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def keepit(ann):
    try:
        keep = ann["iscrowd"] == 0
        return keep
    except:
        return True


def get_crop_box(bboxes, image_info):
    short_size = min(image_info["height"], image_info["width"])
    bbox = bboxes[0]

    if bbox[3] - bbox[1] > short_size or bbox[2] - bbox[0] > short_size:
        return [0, 0, image_info["width"], image_info["height"]]

    center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

    x_left, x_right = center_x - short_size // 2, center_x + short_size // 2
    y_top, y_bottom = center_y - short_size // 2, center_y + short_size // 2

    if x_left < 0:
        x_left, x_right = 0, short_size
    if x_right > short_size:
        x_left, x_right = image_info["width"] - short_size, image_info["width"]

    if y_top < 0:
        y_top, y_bottom = 0, short_size
    if y_bottom > short_size:
        y_top, y_bottom = image_info["height"] - short_size, image_info["height"]

    crop_bbox = [x_left, y_top, x_right, y_bottom]
    return crop_bbox


def pad_to_square(array):
    H, W = array.shape
    max_side = max(H, W)

    padded_array = np.zeros((max_side, max_side), dtype=np.uint8)
    pad_h = (max_side - H) // 2
    pad_w = (max_side - W) // 2
    padded_array[pad_h : pad_h + H, pad_w : pad_w + W] = array

    return padded_array


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def generate_data_list(annotations,image_folder,image_processor,model_config,tokenizer):

    mask_processer = copy.deepcopy(image_processor)
    mask_processer.do_normalize = False
    mask_processer.do_convert_rgb = False
    mask_processer.rescale_factor = 1.0
    
    data_list = []
    for line in annotations:
        img_file = line["filename"]
        bboxs = line["bbox"]
        conv = line["conversations"]
        #input_ids
        prompt = conv[0]["value"]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        # Load image
        #crop_bbox = bboxs[0] #[x_left, y_top, x_right, y_bottom]
        image = Image.open(os.path.join(image_folder, img_file+'.jpg')).convert("RGB")
        print('origin image.size:',image.size)
        image_info = {"height": image.height, "width": image.width}
        #image = image.crop(tuple(crop_bbox))
        images_tensor = process_images([image], image_processor, model_config).unsqueeze(0)
        # make masks
        masks = []
        for bbox in bboxs:
            zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            zero_mask[y1:y2, x1:x2] = 1
            image_aspect_ratio = getattr(model_config, "image_aspect_ratio", None)
            #what is this for?
            #zero_mask = zero_mask[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]]
            if image_aspect_ratio == "pad":
                zero_mask = pad_to_square(zero_mask)
            masks.append(zero_mask)
        masks_pt = []
        for m in masks:
            m = mask_processer.preprocess(m[None, ...], return_tensors="pt")["pixel_values"][0]
            masks_pt.append(m)
        masks = torch.vstack(masks_pt).float().unsqueeze(0)  # (n, h, w) -> (1, n, h, w)
        print('input_ids: ',input_ids.shape)
        print('image_tensor: ',images_tensor.shape)
        print('masks: ',masks.shape)
        print(bboxs)
        data_list.append(
            {
                "id": line["id"],
                "input_ids": input_ids,
                "filename": img_file,
                "conversations": conv,
                "image_tensor": images_tensor,
                "masks": masks,
                "bbox": str(bboxs),
            }
        )

    return data_list

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    annotation_data = json.load(open(args.annotation_file))
    data_list = generate_data_list(annotation_data,args.image_folder,image_processor,model.config,tokenizer)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    for line in tqdm(data_list, total=len(data_list)):
        idx = line["id"]
        image_tensor = line["image_tensor"]
        masks = line["masks"]
        input_ids = line["input_ids"]
        
        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                masks=[masks[0].to(dtype=torch.float16, device="cuda", non_blocking=True)],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        image_id = line["filename"]
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "text": outputs,
                    "gt_name": line["conversations"][1]["value"],
                    "bbox": line["bbox"],
                    "image_id": image_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt_type", type=str, default="bbox")
    parser.add_argument("--erosion", type=bool, default=False)
    parser.add_argument("--dilation", type=bool, default=False)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
