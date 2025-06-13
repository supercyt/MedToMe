#!/bin/bash
set -ex
#  for medmnist_name in bloodmnist
#for medmnist_name in octmnist
for medmnist_name in bloodmnist organamnist dermamnist octmnist tissuemnist
do
  for model_name in vit_tiny vit_small vit_base deit_tiny deit_small deit_base mae_base
  do
    for compress_method in medtome
    do
      /home/caoyitong/miniconda3/envs/pytorch241/bin/python -m compress_test "$model_name" "$medmnist_name" "$compress_method"
    done
  done
done