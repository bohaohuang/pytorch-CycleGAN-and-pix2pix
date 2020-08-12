#!/usr/bin/env bash

cd /home/lab/Documents/bohao/code/mrs
git checkout dev-cyclegan
cd /home/lab/Documents/bohao/code/third_party/pytorch-CycleGAN-and-pix2pix
export PYTHONPATH=$PYTHONPATH:/home/lab/Documents/bohao/code/mrs
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/maps --epoch 50 --name dg_semantic --model cycle_gan \
                                       --direction AtoB --dataset_mode rs --n_epochs 30 --n_epochs_decay 20

cd /home/lab/Documents/bohao/code/mrs
git checkout master
