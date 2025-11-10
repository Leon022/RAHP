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
    ├─image_data.json
    ├─VG-SGG-dicts-with-attri.json
    ├─region_descriptions.json
    ├─vg_cate_dict.json
    └─VG-SGG-with-attri.h5
```

Since GLIP pre-training has seen part of VG150 test images, we remove these images and get new VG150 split and write it to `VG-SGG-with-attri.h5`. 
Please refer to [tools/cleaned_split_GLIPunseen.ipynb](tools/cleaned_split_GLIPunseen.ipynb).

If you are missing some required files (e.g., vg_cate_dict.json), please refer to [https://onedrive.live.com/?id=%2Fpersonal%2F3d84f776196ffd75%2FDocuments%2FRAHP%20file&noAuthRedirect=1] to download or generate them.

### **Relation-Aware & Entity-Aware Prompt Generation Guide**
This script automates the full pipeline of clustering entities into superclasses, validating clusters, generating relation-aware prompts, and converting prompts to the final JSON format.
We also provide pre-generated prompts, please refer to vg_relation_aware_prompts.json and oiv6_relation_aware_prompts.json in the [https://onedrive.live.com/?id=%2Fpersonal%2F3d84f776196ffd75%2FDocuments%2FRAHP%20file&noAuthRedirect=1] directory.

#### Prerequisite: Set OpenAI API Key as Environment Variable
Before running the script, **set your OpenAI API key as an environment variable** (avoids hardcoding keys in commands).  

##### For Linux/macOS Terminal:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Full Automation Script (run.sh for Linux/macOS)
Create a file named `run_prompt_pipeline.sh` with the following content, then execute it via `bash run_prompt_pipeline.sh`.

```bash
#!/bin/bash
set -e  # Exit immediately if any command fails (ensures pipeline integrity)

# -------------------------- Configuration --------------------------
# Update these paths/parameters according to your project structure
DATASET="vg"  # Target dataset (matches your use case: "vg" or "oiv6")
CATE_INFO_PATH="./DATASET/VG150/vg_cate_dict.json"  # Path to entity-category dict
SUPER_ENTITIES_PATH="./DATASET/VG150/vg_super_entities.json"  # Output of Step 1
ENTITY_SUPERCLASS_SAVE_PATH="./DATASET/VG150/vg_entity_superclass_final.json"  # Output of Step 2
REL_PROMPT_OUTPUT_PREFIX="./DATASET/VG150/vg_relation_aware_prompt_"  # Prefix for Step 3 outputs
FINAL_PROMPT_SAVE_PATH="./DATASET/VG150/vg_relation_aware_prompts.json"  # Final output of Step 4

# Clustering & API parameters (adjust if needed)
DISTANCE_THRESHOLD=0.5
LINKAGE_METHOD="ward"
MAX_WORKERS=30
MODEL_NAME="gpt-4o-mini"
RETRY_DELAY=10
# -------------------------------------------------------------------


# -------------------------- Step 1: Cluster Entities into Superclasses --------------------------
echo -e "\n=== Starting Step 1: Cluster Entities into Superclasses ==="
cd tools  # Navigate to the "tools" directory (where your .py scripts are stored)

python cluster_entity_2_super_class.py \
  --dataset "$DATASET" \
  --cate-info-path "$CATE_INFO_PATH" \
  --save-path "$SUPER_ENTITIES_PATH" \
  --distance-threshold "$DISTANCE_THRESHOLD" \
  --linkage-method "$LINKAGE_METHOD"

if [ -f "$SUPER_ENTITIES_PATH" ]; then
  echo "✅ Step 1 Completed: Entity clusters saved to $SUPER_ENTITIES_PATH"
else
  echo "❌ Step 1 Failed: Cluster file not generated"
  exit 1
fi


# -------------------------- Step 2: Validate Superclass Clustering --------------------------
echo -e "\n=== Starting Step 2: Validate Superclass Clustering ==="
# Use the environment variable for OpenAI API key (no hardcoding)
if [ -z "$OPENAI_API_KEY" ]; then
  echo "❌ Error: OPENAI_API_KEY environment variable not set. Set it first (see Prerequisite section)."
  exit 1
fi

python check_super_entity_class.py \
  --openai-api-key "$OPENAI_API_KEY" \
  --super-entities-path "$SUPER_ENTITIES_PATH" \
  --cate-info-path "$CATE_INFO_PATH" \
  --save-path "$ENTITY_SUPERCLASS_SAVE_PATH" \
  --model-name "$MODEL_NAME" \
  --retry-delay "$RETRY_DELAY"

if [ -f "$ENTITY_SUPERCLASS_SAVE_PATH" ]; then
  echo "✅ Step 2 Completed: Validated superclasses saved to $ENTITY_SUPERCLASS_SAVE_PATH"
else
  echo "❌ Step 2 Failed: Validated superclass file not generated"
  exit 1
fi


# -------------------------- Step 3: Generate Relation-Aware Prompts --------------------------
echo -e "\n=== Starting Step 3: Generate Relation-Aware Prompts ==="
python relation_aware_prompt_generation.py \
  --openai-api-key "$OPENAI_API_KEY" \
  --dataset "$DATASET" \
  --output-prefix "$REL_PROMPT_OUTPUT_PREFIX" \
  --max-workers "$MAX_WORKERS" \
  --model-name "$MODEL_NAME" \
  --save-combined

# Verify Step 3 output (check if at least one worker file exists)
FIRST_WORKER_FILE="${REL_PROMPT_OUTPUT_PREFIX}worker_0.jsonl"
if [ -f "$FIRST_WORKER_FILE" ]; then
  echo "✅ Step 3 Completed: Relation-aware prompts saved to ${REL_PROMPT_OUTPUT_PREFIX}worker_*.jsonl"
else
  echo "❌ Step 3 Failed: No prompt files generated"
  exit 1
fi


# -------------------------- Step 4: Convert Prompts to Final JSON --------------------------
echo -e "\n=== Starting Step 4: Convert Prompts to Final JSON ==="
python convert_relation_aware_prompt.py \
  --dataset "$DATASET" \
  --input_prefix "$REL_PROMPT_OUTPUT_PREFIX" \
  --num_workers "$MAX_WORKERS" \
  --output_path "$FINAL_PROMPT_SAVE_PATH"

if [ -f "$FINAL_PROMPT_SAVE_PATH" ]; then
  echo "✅ Step 4 Completed: Final prompts saved to $FINAL_PROMPT_SAVE_PATH"
else
  echo "❌ Step 4 Failed: Final JSON file not generated"
  exit 1
fi


# -------------------------- Pipeline Completion --------------------------
echo -e "\n🎉 All Steps Completed Successfully! Final output: $FINAL_PROMPT_SAVE_PATH"
cd ..  # Return to the parent directory (optional)
```



## Training & Evaluation

### Note
Before running the training script `bash scripts/train.sh`, it is **strongly recommended** to first execute 
```
python tools/generate_relation_aware_embedding.py \
  --dataset-name VG \
  --dataset-dir ./DATASET \
  --clip-backbone ViT-B/32 \
  --save-path ./DATASET/VG150/VG150_relation_aware_embedding.pt
```
to pre-generate the `relation_aware_embedding` file, and set `MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH=relation_aware_embedding_file` in config. This pre-generation step ensures that the training process can directly load the required embedding data, avoiding runtime delays caused by on-the-fly embedding computation and reducing potential training interruptions due to embedding-related issues.

### 1. Training
```
bash scripts/train.sh
```

### 2. Evaluation

```
bash scripts/test.sh
```

## Acknowledgement

This repo is based on [VS3](https://github.com/zyong812/VS3_CVPR23), [PGSG](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024/tree/main), [GLIP](https://github.com/microsoft/GLIP), [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS). Thanks for their contribution.
