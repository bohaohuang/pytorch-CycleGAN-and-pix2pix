#!/usr/bin/env bash

cd /data/users/bh163/code/mrs
git checkout dev-cyclegan
cd /data/users/bh163/code/pytorch-CycleGAN-and-pix2pix
export PYTHONPATH=$PYTHONPATH:/data/users/bh163/code/mrs
CUDA_VISIBLE_DEVICES=5 python train.py --dataroot ./datasets/maps --name syn2dg --model cycle_gan --direction AtoB --dataset_mode rs

cd /data/users/bh163/code/mrs
git checkout master
