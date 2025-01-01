# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pandas as pd

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

import io
import base64
import pickle
from PIL import Image
import math

import torchvision
import torchvision.transforms as T

## data load
def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_pkl_image(image_file):
    images = []
    for i in range(len(image_file)):
        rawbytes = base64.b64decode(image_file[i]) #!!tmp fix for one image case
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB") #load question from .pkl
        images.append(image)

    return images

def load_meta_data(meta_path):
    sample_meta_data, question = [], None
    dict_meta_data = {}
    if meta_path and meta_path!='None':
        file_liat = os.listdir(meta_path)
        for file in file_liat:
            if file.endswith(".txt"): #text type meta data
                with open(os.path.join(meta_path,file),'r') as f:
                    question = f.read().strip('\n').strip()
            elif file.endswith(".jsonl"): #list type meta data
                sample_meta_data += [json.loads(q) for q in open(os.path.join(meta_path,file), "r")]
            elif file.endswith(".json"): #dict type meta data {col_name:{col_value:meta_text}}
                dict_meta_data = json.load(open(os.path.join(meta_path,file), "r"))
        if question!=None:
            print('Override old question with: ')
            print(question)
        if len(sample_meta_data)>0:
            print('Add each sample meta data')
    
    return sample_meta_data, dict_meta_data, question

def get_image_num(image_file):
    if isinstance(image_file,list):
        return len(image_file)
    else:
        return 1

def get_batch(questions, batch_size):
    out_list = [] #(N/B,B)
    questions_len = len(questions)
    start = 0
    while start<questions_len:
        end = start + batch_size
        out_list.append(questions[start:min(end,questions_len)])
        start = end
    return out_list

##data save
def save_jsonl(out_path,data):
    with open(out_path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

## process report
#json to formatted text
def note_dict_to_text(note_dict,list_format=False):
    template = 'There is a {size_w_unit} {characteristic} {density} nodule in the {location} [**Image**].'
    if isinstance(note_dict,str):
        note_dict = json.loads(note_dict.replace('None','"null"'))
    nodule_list = note_dict['nodule_info']
    out_text = []
    for nodule in nodule_list:
        size = nodule['size']
        if (isinstance(size,float) or isinstance(size,int)) and size > 0:
            size_w_unit = '%.1f %s'%(size,nodule['unit'])
        elif isinstance(size,str) and size != 'null':
            size_w_unit = '%s %s'%(size,nodule['unit'])
        else:
            size_w_unit = ''
        characteristic = nodule.get('characteristic','null')
        if characteristic == 'null' or not characteristic:
            characteristic = ''
        density = nodule.get('density','null')
        if density == 'null' or not density:
            density = ''
        location = nodule.get('location','null')
        if location == 'null' or not location:
            location = ''
        #series_images = '(Series %d Image %d)'%(nodule['series'],nodule['image'])
        e_nodule = template.format(size_w_unit=size_w_unit,characteristic=characteristic,density=density,location=location).replace('  ',' ').replace('  ',' ')
        out_text.append(e_nodule)
    if list_format:
        return out_text
    else:
        return ' '.join(out_text)

#process vqa to nodules report json format
def process_vqa_report(ans_data,ans_key='answer'):
    out_data = []
    for name, gp_data in ans_data.groupby('NOTE_ID'):
        note_id = name
        nodule_info = []
        for slice_name,nodule_data in gp_data.groupby('Slice_id'): ### !!! error when slice_id is not unique !!!
            if select_by_qtype(nodule_data,'exist',ans_key).lower() == 'no':
                continue
            nodule_dict = {}
            nodule_dict['density'] = select_by_qtype(nodule_data,['attenuation','density'],ans_key)
            nodule_dict['series'] = int(slice_name.split('_')[1])
            nodule_dict['image'] = int(slice_name.split('_')[2])
            nodule_dict['size'] = select_by_qtype(nodule_data,'size',ans_key)
            if nodule_dict['size'] < 0 or nodule_dict['size'] == 'null' or nodule_dict['series'] < 0 or nodule_dict['image'] < 0:
                continue
            nodule_dict['unit'] = 'mm'
            nodule_dict['location'] = select_by_qtype(nodule_data,'location',ans_key)
            nodule_dict['characteristic'] = select_by_qtype(nodule_data,['characteristic','margin','shape'],ans_key)
            nodule_info.append(nodule_dict)
        #check valid note
        valid_note = 'Yes'
        for nodule in nodule_info:
            if nodule['size'] < 0 or nodule['location'] == 'null':
                valid_note = 'No'
        nodule_json = {'valid_note':valid_note,'nodule_info':nodule_info}
        out_data.append({'NOTE_ID':note_id,'Format_report':nodule_json,'Report_text':note_dict_to_text(nodule_json)})
    return out_data

def select_by_qtype(data,qtypes,ans_key='answer'):  ### !!! error when slice_id is not unique !!!
    if isinstance(qtypes,str):
        qtypes = [qtypes]
    value = data[data['question_type'].isin(qtypes)][ans_key].values
    if len(value) == 0:
        value = 'null'
    else:
        value = value[0]
    if str(value) == 'unable to determine':
        value = 'null'
    if 'size' in qtypes:
        try:
            value = float(value)
        except:
            value = -1
    return value

## main
def eval_model(args):
    #Var
    Use_pkl = False
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    #Dataset
    if args.question_file.endswith('.pkl'):
        with open(os.path.expanduser(args.question_file), "rb") as f:
            data = pickle.load(f)
        questions = data #load question from .pkl
        Use_pkl = True
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.max_samples > 0:
        questions = questions[:args.max_samples]
    question_keys = [k for k in questions[0].keys()]
    if 'image' in question_keys:
        img_key = 'image'
    elif 'filename' in question_keys:
        img_key = 'filename'
    elif "image:" in question_keys:
        img_key = 'image:'
    else:
        img_key = "image"
    batch_size = args.batch_size
    if '40b' in model_path:
        batch_size = 8
    #meta data
    sample_meta_data, dict_meta_data, new_question = load_meta_data(args.meta_path)
    use_meta_prefix = False
    use_meta_postfix = False
    if len(sample_meta_data)>0:
        assert len(sample_meta_data)==len(questions)
        use_meta_prefix = True
        if "question_id" in question_keys:
            meta_data_dict = {n["question_id"]: n["text"] for n in sample_meta_data}
        else:
            meta_data_dict = {i+1: sample_meta_data[i]["text"] for i in range(len(sample_meta_data))}
    if len(dict_meta_data)>0:
        use_meta_postfix = True
        meta_data_dict = {}
        for i,each_data in enumerate(questions):
            if 'question_id' in question_keys:
                idx = each_data["question_id"]
            elif 'image_id' in question_keys:
                idx = each_data["image_id"]
            else:
                idx = i
            col_name = sorted(dict_meta_data)[0]
            meta_data_dict[idx] = dict_meta_data[col_name].get(each_data[col_name],'')

    #Make batchs
    questions_batch = get_batch(questions, batch_size)
    print('Question batchs:')
    print(len(questions_batch))
    answers_file = os.path.expanduser(args.answers_file)
    gt_file = os.path.expanduser(args.answers_file.replace('.json','_gt.json'))
    gt_list = []
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    #check answer file or not
    answer_data = []
    skip_q_id = []
    if not args.rerun and os.path.isfile(answers_file):
        with open(answers_file, "r") as f:
            answer_data = [json.loads(n) for n in f]
            skip_q_id = [n["question_id"] for n in answer_data]
            print('Already have %d answers'%len(skip_q_id))
    ans_file = open(answers_file, "w")
    i = 0
    for batch in tqdm(questions_batch):
        idx_list = []
        image_tensor_list = []
        prompt_list = []
        input_ids_list = []
        for line in batch:
            i += 1
            if 'question_id' in question_keys:
                idx = line["question_id"]
            elif 'image_id' in question_keys:
                idx = line["image_id"]
            else:
                idx = i
            
            image_file = line[img_key]
            #tmp fix, transport to list
            if '[' in image_file and ']' in image_file and ', ' in image_file:
                image_file = image_file.strip('[').strip(']').split(', ')
            #load qs and save gt
            if Use_pkl:
                qs = line['question']
                gt_dict = {
                    "question_id": idx,
                    'prompt': qs,
                    'answer': line['answer']
                }
                gt_list.append(gt_dict)
            else:
                qs = line["text"]
            #chnage qs if need
            if new_question!=None:
                if args.replace_q:
                    qs = new_question
                else:
                    qs = new_question + '\n' + qs
            
            if idx in skip_q_id:
                ans_idx = skip_q_id.index(idx)
                ans_dict = answer_data[ans_idx]
                ans_file.write(json.dumps(ans_dict) + "\n")
                ans_file.flush()
                continue

            #add meta data if have
            if use_meta_prefix:
                sample_meta = meta_data_dict[idx]
                qs = sample_meta + '\n' + qs
                if i==1:
                    print('Add meta data to question')
                    print(qs)
            if use_meta_postfix:
                sample_meta = meta_data_dict[idx]
                qs = qs + '\n' + sample_meta
                if i==1:
                    print('Add meta data to question')
                    print(qs)
            #concat with image (add multi image case)
            cur_prompt = qs
            image_num = get_image_num(image_file)
            for im_i in range(image_num):
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            if Use_pkl:
                images = load_pkl_image(image_file)
            else:
                if isinstance(image_file,list):
                    images = [load_image(os.path.join(args.image_folder, e_image_file)) for e_image_file in image_file]
                else:
                    images = [load_image(os.path.join(args.image_folder, image_file))]
            
            #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            #store to list
            idx_list.append(idx)
            image_tensor_list.append(image_tensor)
            prompt_list.append(cur_prompt)
            input_ids_list.append(input_ids)
        
        if len(idx_list)==0:
            continue
        
        #batch part
        input_ids_tensor = torch.cat(input_ids_list)
        image_tensors = torch.cat(image_tensor_list)

        with torch.inference_mode():
            output_ids_list = model.generate(
                input_ids_tensor,
                images=[image_tensors],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs_batch = tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        

        for _i in range(len(idx_list)):
            idx = idx_list[_i]
            cur_prompt = prompt_list[_i]
            outputs = outputs_batch[_i]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
    
    #save gt if need
    if len(gt_list)>0:
        save_jsonl(gt_file, gt_list)
    ans_file.close()
    
    #process report if need
    req_cols = ['NOTE_ID','Slice_id','question_type']
    if set(req_cols) <= set(question_keys):
        with open(answers_file, "r") as f:
            answer_data = [json.loads(n) for n in f]
        ans_data = pd.DataFrame(answer_data)
        q_data = pd.DataFrame(questions).drop(columns=['text'])
        full_data = pd.merge(q_data, ans_data, on='question_id')
        report_data = process_vqa_report(full_data,ans_key='text')
        report_file = os.path.expanduser(answers_file.replace('.jsonl','_fmtreport.jsonl'))
        save_jsonl(report_file, report_data)
    else:
        print('Question keys need:',req_cols)
        print('Question keys:',question_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--meta-path", type=str, default=None,help='metadata path .jsonl with each sample. .txt for all sample(will override origin qs)')
    parser.add_argument("--replace-q", action='store_true')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--rerun", action='store_true')
    args = parser.parse_args()

    eval_model(args)
