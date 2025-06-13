import glob
import json
import logging
import os
import sys

from sympy.stats import quantile
from thop import profile, clever_format

import medmnist
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from medmnist.evaluator import getACC, getAUC
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import pretrain_model_config, MedMNIST_DATA_ROOT, f1
from algo import medtome

device = "cuda:0"
model_name = sys.argv[1]
# bloodmnist dermamnist octmnist organamnist pathmnist tissuemnist
data_flag = sys.argv[2]
compress_method = sys.argv[3]

# model_name = "deit_tiny"
# # bloodmnist dermamnist octmnist organamnist pathmnist tissuemnist
# data_flag = "bloodmnist"
# compress_method = "kmtome"

compress = globals()[compress_method]


size = 224

BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
labels = info['label']
pretrain_model_path = pretrain_model_config[model_name]["model_path"]
model_tag = pretrain_model_config[model_name]["model_tag"]
norm_mean = pretrain_model_config[model_name]["mean"]
norm_std = pretrain_model_config[model_name]["std"]

# define loss function and optimizer
normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

DataClass = getattr(medmnist, info['python_class'])

test_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    normalize
])

target_transform = lambda x: x[0] if len(x) == 1 else x

test_dataset = DataClass(split='test', as_rgb=True, size=size,
                         transform=test_transform, root=MedMNIST_DATA_ROOT,
                         target_transform=target_transform,  mmap_mode='r')


data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

def test_tome_model(model_path, ratio=.0):
    model = timm.create_model(model_tag, num_classes=n_classes, checkpoint_path=model_path).to(device)
    if model_tag in ["lvvit_s", "lvvit_m"]:
        skip_lam = 2
    else:
        skip_lam = 1
    compress.patch.deit(model, skip_lam=skip_lam)
    # loading the pre-trained model
    logging.info(f"ratio: {ratio}")
    if compress_method == "DiffRate":
        model.init_kept_num_using_ratio(ratio)
    else:
        model.ratio = ratio

    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.softmax(dim=-1)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            y_true = torch.cat((y_true, targets.cpu()), 0)

        y_score = y_score.detach().numpy()
        y_true = y_true.detach().numpy()

        f1_score = f1.compute(predictions=np.argmax(y_score, axis=1), references=y_true, average="macro")
        accuracy_score = getACC(y_true, y_score, task)
        auc_score = float(getAUC(y_true, y_score, task))

        x = torch.randn(1, 3, 224, 224).cuda()

        flops, params = profile(model, inputs=(x,))
        flops, params = clever_format([flops, params], '%.6f')
        # benchmark(model, device=torch.device(device), verbose=True)
        print('flops', flops)
        print('params', params)

    return {**f1_score, "auc_score": auc_score, "accuracy_score": accuracy_score}, flops


metric_result = {}
os.makedirs(f"test_metrics/{compress_method}_metrics", exist_ok=True)

finetuing_model=f"./MedMNIST_ft_model/{model_name}/{model_name}-{data_flag}"
best_checkpoint = glob.glob(f"{finetuing_model}/best_*.pt")[0]
for idx, ratio in enumerate([1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.86]):
    metrics, flops = test_tome_model(best_checkpoint, ratio=ratio)
    metric_result[ratio] = metrics
    metric_result[ratio]["flops"] = flops

with open(f"test_metrics/{compress_method}_metrics/{model_name}-{data_flag}.json", "w") as file:
    json.dump(metric_result, file, ensure_ascii=False, indent=2)
