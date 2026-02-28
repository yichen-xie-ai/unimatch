#!/usr/bin/env bash

# basic GMDepth without any refinement (1/8 feature only)

# number of gpus for training, please set according to your hardware
<<<<<<< Updated upstream
# trained on 3x 80GB A100 gpus
=======
# trained on 3 x 80GB A100 gpus
>>>>>>> Stashed changes
NUM_GPUS=3


# # scannet (our final model is trained for 100K steps, for ablation, we train for 50K)
# # resume flow things model (our ablations are trained from random init)
# CHECKPOINT_DIR=checkpoints_depth/scannet-gmdepth-scale1-resumeflowthings && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_depth.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume pretrained/gmflow-scale1-things-e9887eda.pth \
# --no_resume_optimizer \
# --dataset scannet \
# --val_dataset scannet \
# --image_size 480 640 \
# --batch_size 80 \
# --lr 4e-4 \
# --summary_freq 100 \
# --val_freq 5000 \
# --save_ckpt_freq 5000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# demon, resume flow things model
DATASET=scenes11
CHECKPOINT_DIR=checkpoints_depth/demon-gmdepth-scale1_${DATASET}_bs78-urope_H2-woSine && \
mkdir -p ${CHECKPOINT_DIR} && \
NCCL_P2P_DISABLE=1 torchrun \
--nproc_per_node=${NUM_GPUS} \
--master_port=29500 \
main_depth.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--no_resume_optimizer \
--dataset demon \
--val_dataset demon \
--demon_split ${DATASET} \
--image_size 448 576 \
--batch_size 78 \
--lr 4e-4 \
--summary_freq 100 \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 100000 \
--workers 8 \
--attn_type swin \
--rope_type urope \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


