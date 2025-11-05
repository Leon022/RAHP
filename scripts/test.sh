#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
echo "Testing!!!!"
MODEL_NAME="SGDet-set-RAHP-open-vocabulary-relation"
MODEL_CK="0040000"

python -m torch.distributed.launch \
    --master_port 8888 --nproc_per_node=${NUM_GPUS} \
    tools/test_grounding_net.py \
    --task_config configs/vg150/finetune.yaml \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    DATASETS.VG150_OPEN_VOCAB_MODE False \
    MODEL.WEIGHT ./OUTPUT/${MODEL_NAME}/model_${MODEL_CK}.pth \
    OUTPUT_DIR ./OUTPUT/TEST-${MODEL_NAME}
