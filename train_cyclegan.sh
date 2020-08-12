#!/usr/bin/env bash

A_DIR=/data/users/bh163/data/mrs/synthinel_v205_random/building/ps512_pd0_ol0/
B_DIR=/data/users/bh163/data/mrs/deepglobe/14p_pd0_ol0
CITY_NAME=Shanghai
A_APPENDIX=_base
SEMANTIC_DIR=/data/users/bh163/models/mrs/syn_104/inria/ecresnet50_dcunet_dsinria_lre1e-03_lrd1e-02_ep80_bs6_ds50_dr0p1

cd /data/users/bh163/code/mrs
git checkout dev-cyclegan
cd /data/users/bh163/code/pytorch-CycleGAN-and-pix2pix
export PYTHONPATH=$PYTHONPATH:/data/users/bh163/code/mrs
python train.py --dataroot ./datasets/maps \
                --name syn205_random_dg_${CITY_NAME} \
                --model cycle_gan \
                --a_dir ${A_DIR} \
                --b_dir ${B_DIR} \
                --a_appendix ${A_APPENDIX} \
                --city_name ${CITY_NAME} \
                --semantic_dir ${SEMANTIC_DIR} \
                --direction AtoB \
                --dataset_mode rs \
                --display_id 0 \
                --n_epochs 30 \
                --n_epochs_decay 10 \
                --batch_size 6 \
                --load_size 512 \
                --crop_size 512 \
                --gpu_ids 0,1,2,3,4,5 \

cd /data/users/bh163/code/mrs
git checkout master
