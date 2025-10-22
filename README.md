# Official Implementation of "Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation"

## Introductionp
Our paper ["Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation"](https://arxiv.org/abs/2412.19021) AAAI 2025.


## Installation and Setup

***Environment.***
This repo requires Pytorch>=1.9 and torchvision.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo 
pip install transformers openai
pip install SceneGraphParser spacy 
python setup.py build develop --user
```

***Pre-trained Visual-Semantic Space.*** Download the pre-trained `GLIP-T` and `GLIP-L` [checkpoints](https://github.com/microsoft/GLIP#model-zoo) into the ``MODEL`` folder. 
(!! GLIP has updated the downloading paths, please find these checkpoints following https://github.com/microsoft/GLIP#model-zoo)
```
mkdir MODEL
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O swin_tiny_patch4_window7_224.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O swin_large_patch4_window12_384_22k.pth
```

## Dataset Preparation

1. Visual Genome
* ``Visual Genome (VG)``: Download the original [VG](https://visualgenome.org/) data into ``DATASET/VG150`` folder. Refer to [vg_prepare](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md).

2. Openimage V6
* ``Openimage V6``: 
    1. The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).

    2. The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 

    3. You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3)
    4. By unzip the downloaded datasets, the dataset dir contains the `images` and `annotations` folder. 
    Link the `open-imagev6` dir to the `./cache/openimages` then you are ready to go.
    ```bash
    mkdir datasets/openimages
    ln -s /path/to/open_imagev6 datasets/openimages ./cache/cache
    ```


The `DATASET` directory is organized roughly as follows:
```
├─Openimage V6
│  ├─annotations
│  ├─images
└─VG150
    ├─VG_100K
    ├─weak_supervisions
    ├─image_data.json
    ├─VG-SGG-dicts-with-attri.json
    ├─region_descriptions.json
    └─VG-SGG-with-attri.h5 
```

Since GLIP pre-training has seen part of VG150 test images, we remove these images and get new VG150 split and write it to `VG-SGG-with-attri.h5`. 
Please refer to [tools/cleaned_split_GLIPunseen.ipynb](tools/cleaned_split_GLIPunseen.ipynb).

### **Relation-Aware & Entity-Aware Prompt Generation Guide**

To generate relation-aware prompts and entity-aware prompts, please follow these three steps sequentially:

#### **Step 1: Cluster Entity Types into Superclasses**
```bash
cd tools
python cluster_entity_2_super_class.py
```
*This will group entity types into hierarchical superclasses.*

#### **Step 2: Validate Superclass Clustering**
```bash
python check_super_entity_class.py
```
*Verifies the quality of generated superclasses before prompt generation.*

#### **Step 3: Generate Relation-Aware Prompts**
```bash
python relation_aware_prompt_generation.py
```
*Produces final prompts incorporating both entity hierarchies and relation contexts.*



## Training & Evaluation

1. Training
```
bash scripts/train.sh
```

2. Evaluation

```
bash scripts/test.sh
```

## Acknowledgement

This repo is based on [VS3](https://github.com/zyong812/VS3_CVPR23), [PGSG](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024/tree/main), [GLIP](https://github.com/microsoft/GLIP), [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS). Thanks for their contribution.
