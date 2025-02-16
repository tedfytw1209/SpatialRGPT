import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    depth_path: str = field(default=None, metadata={"help": "Path to the training depth data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/PATH/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/PATH/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)

    llava_1_5_sft = Dataset(
        dataset_name="llava_1_5_sft",
        dataset_type="torch",
        data_path="/PATH/llava_v1_5_mix665k.json",
        image_path="/PATH/data",
    )
    add_dataset(llava_1_5_sft)

    spatialrgpt_ft = Dataset(
        dataset_name="spatialrgpt_ft",
        dataset_type="spatialrgpt",
        data_path="/orange/bianjiang/tienyu/OpenSpatialDataset/result_10_depth_convs.json",
        image_path="/PATH/Openimages/train",
        depth_path="/PATH/relative_depth/raw",
        description="900K SFT data by SpatialRGPT (submission) w/ depth (template+LLaMa rephrased).",
    )
    add_dataset(spatialrgpt_ft)
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
    #region ver. LDCT
    ldct_vqa_f_reg = Dataset(
        dataset_name='ldct_vqa_f_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CT/ldctf_region.json',
        image_path='/orange/bianjiang/tienyu/IRB202400720_CT/',
        depth_path='',
        description='LDCT VQA Filtered Data (Region).'
    )
    add_dataset(ldct_vqa_f_reg)
    ldct2_vqa_f_reg = Dataset(
        dataset_name='ldct2_vqa_f_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CTv2/ldct2f_norm_region.json',
        image_path='/orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/',
        depth_path='',
        description='LDCT2 VQA Filtered Data (Region) with normalizations.'
    )
    add_dataset(ldct2_vqa_f_reg)
    ldct2_vqa_wunk_reg = Dataset(
        dataset_name='ldct2_vqa_wunk_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CTv2/ldct2f_wunk_region.json',
        image_path='/orange/bianjiang/tienyu/IRB202400720_LDCT2_norm/',
        depth_path='',
        description='LDCT2 VQA Filtered Data (Region) with normalizations.'
    )
    add_dataset(ldct2_vqa_wunk_reg)
    
    #region ver. LCCT
    lcct_vqa_f_reg = Dataset(
        dataset_name='lcct_vqa_f_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CT/lcctf_region_train.json',
        image_path='/orange/bianjiang/tienyu/IRB202201140_Lung_CT2/',
        depth_path='',
        description='LCCT VQA Filtered Data (Region).'
    )
    add_dataset(lcct_vqa_f_reg)
    lcct2_vqa_f_reg = Dataset(
        dataset_name='lcct2_vqa_f_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CTv2/lcct2f_norm_region.json',
        image_path='/orange/bianjiang/tienyu/IRB202201140_Lung_CT2_norm/',
        depth_path='',
        description='LCCT VQA Filtered Data (Region).'
    )
    add_dataset(lcct2_vqa_f_reg)
    lcct2_vqa_wunk_reg = Dataset(
        dataset_name='lcct2_vqa_wunk_reg',
        dataset_type='spatialrgpt',
        data_path='/orange/bianjiang/tienyu/SpatialRGPT/data/LungCancer_CTv2/lcct2f_wunk_region.json',
        image_path='/orange/bianjiang/tienyu/IRB202201140_Lung_CT2_norm/',
        depth_path='',
        description='LCCT VQA Filtered Data (Region).'
    )
    add_dataset(lcct2_vqa_wunk_reg)
    
    #non region ver.
    ldct_vqa_f = Dataset(
        dataset_name='ldct_vqa_f',
        dataset_type='vflan',
        data_path='/orange/bianjiang/tienyu/LC_CT/vqa2_f_data'
    )
    add_dataset(ldct_vqa_f)
    lcct_vqa_f = Dataset(
        dataset_name='lcct_vqa_f',
        dataset_type='vflan',
        data_path='/orange/bianjiang/tienyu/LC_CT/vqa_f_data/'
    )
    add_dataset(lcct_vqa_f)
    
    
