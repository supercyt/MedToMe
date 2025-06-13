import numpy as np
from evaluate import load
from medmnist.evaluator import getACC, getAUC

PRETRAINED_MODEL_ROOT = ""
MedMNIST_DATA_ROOT = ""

pretrain_model_config = {
    "vit_tiny": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/vit_tiny_patch16_224.augreg_in21k.npz",
        "model_tag": "vit_tiny_patch16_224.augreg_in21k",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
    "vit_small": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/vit_small_patch16_224.augreg_in21k.npz",
        "model_tag": "vit_small_patch16_224.augreg_in21k",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
    "vit_base": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/vit_base_patch16_224.augreg_in21k.npz",
        "model_tag": "vit_base_patch16_224.augreg_in21k",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
    "deit_base": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/deit_base_patch16_224-b5f2ef4d.pth",
        "model_tag": "deit_base_patch16_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "deit_small": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/deit_small_patch16_224-cd65a155.pth",
        "model_tag": "deit_small_patch16_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "deit_tiny": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/deit_tiny_patch16_224-a1311bcf.pth",
        "model_tag": "deit_tiny_patch16_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "mae_base": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/deit_tiny_patch16_224-a1311bcf.pth",
        "model_tag": "vit_base_patch16_224.mae",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "swin_base": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/swin_base_patch4_window7_224_22k.pth",
        "model_tag": "swin_base_patch4_window7_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "swin_small": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/swin_small_patch4_window7_224_22k.pth",
        "model_tag": "swin_small_patch4_window7_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "swin_tiny": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/swin_tiny_patch4_window7_224_22k.pth",
        "model_tag": "swin_tiny_patch4_window7_224",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "lvvit_tiny": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/lvvit_t.pth",
        "model_tag": "lvvit_t",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "lvvit_small": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/lvvit_s-26M-224-83.3.pth.tar",
        "model_tag": "lvvit_s",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "lvvit_base": {
        "model_path": f"{PRETRAINED_MODEL_ROOT}/lvvit_m-56M-224-84.0.pth.tar",
        "model_tag": "lvvit_m",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}


# accuracy = load("metrics/accuracy")
# roc_auc = load("metrics/roc_auc", "multiclass")
f1 = load("metrics/f1")
