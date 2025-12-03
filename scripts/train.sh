#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
export DATASET_DIR="$(realpath "${SCRIPT_DIR}/../DATASET")"
echo ${DATASET_DIR}

export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=4

echo "TRAINING OV SGDet"

MODEL_NAME='SGDet-set-RAHP-open-vocabulary-relation'
mkdir -p ./OUTPUT/${MODEL_NAME}/
cp ./tools/train_net.py ./OUTPUT/${MODEL_NAME}/
cp ./maskrcnn_benchmark/engine/trainer.py ./OUTPUT/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/detector/generalized_vl_rcnn.py ./OUTPUT/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/rpn/vldyhead.py ./OUTPUT/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/relation_head/ov_classifier.py ./OUTPUT/${MODEL_NAME}
cp ./scripts/train.sh ./OUTPUT/${MODEL_NAME}/

python3 -m torch.distributed.launch \
    --master_port 8886 --nproc_per_node=${NUM_GPUS} \
    tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 8*${NUM_GPUS} )) \
    TEST.IMS_PER_BATCH $(( 8*${NUM_GPUS}  )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 3000 \
    DATASETS.VG150_OPEN_VOCAB_MODE False \
    OUTPUT_DIR ./OUTPUT/${MODEL_NAME} \
    MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH ../DATASET/VG150/VG150_relation_aware_embedding.pt


