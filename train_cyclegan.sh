#!/usr/bin/env bash

cd /data/users/bh163/code/mrs
git checkout dev-cyclegan
cd /data/users/bh163/code/pytorch-CycleGAN-and-pix2pix
export PYTHONPATH=$PYTHONPATH:/data/users/bh163/code/mrs
python train.py --dataroot ./datasets/maps --name syn205_random_dg_full --model cycle_gan --direction AtoB --dataset_mode rs --display_id 0 --n_epochs 30 --n_epochs_decay 10 --batch_size 6 --load_size 512 --crop_size 512 --gpu_ids 0,1,2,3,4,5

cd /data/users/bh163/code/mrs
git checkout master
