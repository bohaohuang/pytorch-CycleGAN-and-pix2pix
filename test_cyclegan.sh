#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /hdd/style_transfer/source/inria/ \
                                      --num_test 2500 \
                                      --name inria2dg \
                                      --model test \
                                      --epoch latest \
                                      --load_size 512 \
                                      --crop_size 512 \
                                      --no_dropout \
                                      --checkpoints_dir /hdd6/Models/style_transfer/ \
                                      --results_dir /hdd/style_transfer/target/inria2dg_2