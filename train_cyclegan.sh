#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/data/users/bh163/code/mrs
CUDA_VISIBLE_DEVICES=5 python train.py --dataroot ./datasets/maps --name syn2dg --model cycle_gan --direction AtoB --dataset_mode rs
