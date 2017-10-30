#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --batch-size, --gpus.

## train se-inception-v4
# LD_PRELOAD="/usr/lib/libtcmalloc.so" python -u train_se_inception_v4.py --data-dir data/VGG_DATA --data-type vggface --batch-size 384 --num-classes 2613 --num-examples 1484577 --gpus=6,7,8,9
python -u train_se_inception_v4.py --data-dir data/imagenet --data-type imagenet --batch-size 384 --gpus=6,7,8,9