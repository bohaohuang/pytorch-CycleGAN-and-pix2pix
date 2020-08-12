#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /hdd/style_transfer/syn205_random \
                                      --num_test 12288 \
                                      --name syn205_random_inria \
                                      --model test \
                                      --epoch latest \
                                      --no_dropout \
                                      --checkpoints_dir /home/lab/Documents/bohao/code/third_party/pytorch-CycleGAN-and-pix2pix/checkpoints \
                                      --results_dir /hdd/style_transfer/inria_syn205