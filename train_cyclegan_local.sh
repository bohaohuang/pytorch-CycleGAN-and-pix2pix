#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/maps --epoch 50 --name [model_name] --model cycle_gan \
                                       --direction AtoB --dataset_mode unaligned --n_epochs 30 --n_epochs_decay 20 \
                                       --init_weights None
