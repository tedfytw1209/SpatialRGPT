# #!/usr/bin/env python
import argparse
import json
import os
import base64
import pickle
from collections import Counter
from tqdm import tqdm
import copy
from PIL import Image
from io import BytesIO
import pandas as pd

def encode_image_to_base64(image_path,resize=-1,img_format='PNG'):
    if resize>0:
        im = Image.open(image_path)
        if resize > 1:
            width, height = im.size
            min_edge = min(width,height)
            resize = min_edge / resize
            new_size = (int(width/resize), int(height/resize))
            #new_size = (int(resize), int(resize))
        else:
            width, height = im.size
            new_size = (int(width*resize), int(height*resize))
        im = im.resize(new_size)
        buffered = BytesIO()
        im.save(buffered, format=img_format)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        with open(image_path, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_string

def save_dataset(dataset_type, dataset_name, save_path, data):
    save_filename = f"{dataset_type}_{dataset_name}.pkl"
    save_pathname = os.path.join(save_path, save_filename)
    with open(save_pathname, "wb") as f:
        pickle.dump(data, f)

def get_bbox_list(bbox_data):
    #mask[y:y+rect_height,x:x+rect_width] = 255
    out_list = []
    for i,row in bbox_data.iterrows():
        try:
            x = int(row['X'])
            y = int(row['Y'])
            rect_width = int(row['Width'])
            rect_height = int(row['Height'])
            out_list.append([x,y,x+rect_width,y+rect_height])
        except:
            pass
    #error handling
    if len(out_list)==0:
        out_list = [[0,0,512,512]]
    return out_list

parser = argparse.ArgumentParser(description="Region Dataset Creation")
parser.add_argument(
    "-i",
    "--image-path",
    default="",
    help="dir for input image",
)
parser.add_argument(
    "-t",
    "--txt-path",
    default="",
    help="file for question input",
)
parser.add_argument(
    "-a",
    "--answer-path",
    default="",
    help="file for answer input",
)
parser.add_argument(
    "-b",
    "--bbox-path",
    default="",
    help="file for bbox input",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default="",
    help="dir for output",
)
parser.add_argument(
    "-n",
    "--name",
    default="iuxray_w_pred",
    help="name for output",
)

args = parser.parse_args()

image_dir = args.image_path
output_dir = args.output_dir
ans_path = args.answer_path
txt_path = args.txt_path
bbox_path = args.bbox_path
bal_sam_method = ''

'''Format:
[
    {
        "id": <question_id>,
        'filename': <name without .jpg>,
        "conversations": <conversation text>,
        "rle" / "segmentation" / "bbox": <mask:[x1, y1, x2, y2]>,
    }
]
'''

with open(txt_path, 'r') as json_file:
    json_list = list(json_file)

with open(ans_path, 'r') as json_file:
    ans_json_list = list(json_file)

ans_jsons = [json.loads(_item) for _item in ans_json_list]
filepaths = [json.loads(_item) for _item in json_list]
if len(ans_jsons)==1:
    ans_jsons = ans_jsons[0]
if len(filepaths)==1:
    filepaths = filepaths[0]
print('ans len: ',len(ans_jsons))
print('question len:',len(filepaths))
assert len(ans_jsons)==len(filepaths)
num_cases = len(filepaths)
bbox_data = pd.read_csv(bbox_path)

if bal_sam_method:
    #count answers
    dup_times = []
    ref_list = [n['text'] for n in ans_jsons]
    ref_count = Counter(ref_list)
    ref_count_list = [n for n in ref_count.values()]
    ref_count_max = min(max(ref_count_list),5)
    dup_times = [max(1,int(ref_count_max/ref_count[n])) for n in ref_list]
    print(dup_times)
else:
    dup_times = [1 for i in range(len(ans_jsons))]

data_dict = []
test_dict = []
data_num = 0
if 'image' in filepaths[0]:
    img_key = 'image'
elif 'filename' in filepaths[0]:
    img_key = 'filename'

if 'text' in filepaths[0]:
    qs_key = 'text'
elif 'question' in filepaths[0]:
    qs_key = 'question'
    
#image_id
if 'image_id' in filepaths[0]:
    id_key = 'image_id'
elif 'question_id' in filepaths[0]:
    id_key = 'question_id'

default_qs = "Describe the image in detail."

for _i in tqdm(range(num_cases)):
    txt_results = filepaths[_i]
    ans_results = ans_jsons[_i]
    image_files = filepaths[_i][img_key]
    note_id = txt_results['NOTE_ID']
    bbox_data_e = bbox_data[bbox_data['Image']==image_files]
    n_mask = len(bbox_data_e)
    #tmp fix, transport to list
    if image_files.endswith('.jpg'):
        img_list = image_files.replace('.jpg','')
    else:
        img_list = image_files
    reference_report = ans_results['text']
    if not reference_report or reference_report=='None' or reference_report==None:
        print('No ground truth answer, skip')
        exit()
    qs_id = txt_results[id_key]
    #bounding box
    bbox_list = get_bbox_list(bbox_data_e)
    #qs_txt to conversation
    question = txt_results.get(qs_key,default_qs)
    question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
    question = "<image>\n" + question + 'in' + '<mask>'*n_mask
    conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": reference_report},
            ]
    #save to new json
    for d in range(dup_times[_i]):
        _dict = {
        "id": qs_id,
        "conversations": conversation,
        "filename": img_list,
        "bbox": bbox_list,
        }
        data_dict.append(_dict)
    
out_path = os.path.join(output_dir,args.name+'.json')
with open(out_path, 'w') as outfile:
    json.dump(data_dict, outfile)