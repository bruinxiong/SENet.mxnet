#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.

## train se-resnext-50
python -u train_se_resnext_w_d.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 192 --num-group 64 --drop-out 0.0 --gpus=6,7,8,9

## train se-resnext-101
#python -u train_se_resnext_w_d.py --data-dir data/imagenet --data-type imagenet --depth 101 --batch-size 128 --num-group 64 --drop-out 0.0 --gpus=2,3,4,5

## train se-resnet-50
#python -u train_se_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 256 --gpus=6,7,8,9

## train se-resnet-101
#python -u train_se_resnet.py --data-dir data/imagenet --data-type imagenet --depth 101 --batch-size 256 --gpus=2,3,4,5
