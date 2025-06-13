import glob
import json
import logging
import os
import sys

import medmnist
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from medmnist.evaluator import getACC, getAUC
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from algo import medtome

from config import pretrain_model_config, MedMNIST_DATA_ROOT, f1

device = "cuda:0"
model_name = sys.argv[1]
# bloodmnist dermamnist octmnist organamnist pathmnist tissuemnist
data_flag = sys.argv[2]
compress_method = sys.argv[3]
ratio = float(sys.argv[4])
size = 224

finetuing_model=f"./retrain_result/{compress_method}/{model_name}/{model_name}-{data_flag}"
model_path = glob.glob(f"{finetuing_model}/best_*.pt")[0]

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


model = timm.create_model(model_tag, num_classes=n_classes).to(device)
# loading the pre-trained model
if model_tag in ["lvvit_s", "lvvit_m"]:
    skip_lam = 2
else:
    skip_lam = 1
compress = globals()[compress_method]
compress.patch.deit(model, skip_lam=skip_lam)
# loading the pre-trained model
logging.info(f"ratio: {ratio}")
if compress_method == "DiffRate":
    model.init_kept_num_using_ratio(ratio)
else:
    model.ratio = ratio

load_checkpoint(model, model_path)

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

    metric_result = {**f1_score, "auc_score": auc_score, "accuracy_score": accuracy_score}

    os.makedirs(f"retrain_test_metrics/{compress_method}_metrics", exist_ok=True)
    with open(f"retrain_test_metrics/{compress_method}_metrics/{model_name}-{data_flag}.json", "w") as file:
        json.dump(metric_result, file, ensure_ascii=False)