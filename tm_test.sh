#!/bin/bash
set -ex
for model_name in vit_base deit_base mae_base deit_small vit_small deit_tiny vit_tiny
  do
    for medmnist_name in bloodmnist dermamnist octmnist organamnist tissuemnist
    do
      for compress_method in medtome
      do
        /home/caoyitong/miniconda3/envs/pytorch241/bin/python -m tm_timm_test "$model_name" "$medmnist_name" "$compress_method" "0.86"
      done
    done
done