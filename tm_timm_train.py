import glob
import logging
import os
import sys

import medmnist
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import transformers
from medmnist import INFO
from medmnist.evaluator import getACC, getAUC
from tqdm import tqdm
from algo import medtome

from config import MedMNIST_DATA_ROOT, f1, pretrain_model_config

logging.basicConfig(level=logging.INFO)

def delete_last_model(model_dir, symbol):
    last_model = glob.glob(f"{model_dir}/{symbol}*.pt")
    if len(last_model) != 0:
        os.remove(last_model[0])

def save_new_model_and_delete_last(model, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    if delete_last_model is not None:
        delete_last_model(save_dir, delete_symbol)

    torch.save(model.state_dict(), save_path)
    logging.info(f"\nmodel is saved in {save_path}")


device = "cuda:0"
model_name = sys.argv[1]
data_flag = sys.argv[2]
compress_method = sys.argv[3]
ratio = float(sys.argv[4])
size = 224

NUM_EPOCHS = 3
save_steps = 1.1
eval_steps = 0.1
warmup_steps = 0.1

output_dir=f"./retrain_result/{compress_method}/{model_name}/{model_name}-{data_flag}"
best_model = glob.glob(f"{output_dir}/best_model_*")
if len(best_model) > 0:
    logging.info(f"model {best_model[0]} has existed")
    sys.exit(0)

if compress_method == "DiffRate":
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 128
lr = 1e-5
weight_decay = 1e-6

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
labels = info['label']

model_tag = pretrain_model_config[model_name]["model_tag"]
norm_mean = pretrain_model_config[model_name]["mean"]
norm_std = pretrain_model_config[model_name]["std"]
finetuing_model=f"./MedMNIST_ft_model/{model_name}/{model_name}-{data_flag}"
model_path = glob.glob(f"{finetuing_model}/best_*.pt")[0]
# define loss function and optimizer
model = timm.create_model(model_tag, num_classes=n_classes, checkpoint_path=model_path).to(device)

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

normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.AugMix(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    normalize
])

target_transform = lambda x: x[0] if len(x) == 1 else x

train_dataset = DataClass(split='train', as_rgb=True, size=size,
                          transform=train_transform, target_transform=target_transform,
                          root=MedMNIST_DATA_ROOT, mmap_mode='r')
val_dataset = DataClass(split='val', as_rgb=True, size=size,
                        transform=test_transform, root=MedMNIST_DATA_ROOT,
                        target_transform=target_transform, mmap_mode='r')

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

batches_per_epoch = len(train_loader)
max_steps = NUM_EPOCHS * batches_per_epoch

warmup_steps = int(warmup_steps * max_steps) if warmup_steps < 1 else warmup_steps
eval_steps = int(max_steps * eval_steps) if eval_steps < 1 else eval_steps
save_steps = int(max_steps * save_steps) if save_steps < 1 else save_steps

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_training_steps=max_steps, num_warmup_steps=warmup_steps)

best_f1_score = .0
train_iter = iter(train_loader)

with tqdm(total=max_steps) as process_bar:
    # forward + backward + optimize
    for step in range(max_steps):
        process_bar.set_description('Step %i' % step)
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)  # 重置迭代器
            inputs, targets = next(train_iter)

        optimizer.zero_grad()
        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        process_bar.set_postfix(loss=loss.item(), lr=lr)

        if step == 0 or (step + 1) % eval_steps == 0 or (step + 1) % save_steps == 0:
            model.eval()
            y_score = torch.Tensor([])
            y_gt = torch.Tensor([])

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    outputs = outputs.softmax(dim=-1)
                    y_score = torch.cat((y_score, outputs.cpu()), 0)
                    y_gt = torch.cat((y_gt, targets.cpu()), 0)

                y_score = y_score.detach().numpy()
                y_gt = y_gt.detach().numpy()

                f1_score = f1.compute(predictions=np.argmax(y_score, axis=1), references=y_gt, average="macro")["f1"]
                acc = getACC(y_gt, y_score, task)
                auc = float(getAUC(y_gt, y_score, task))
                logging.info(f"f1:{f1_score}, acc: {acc}, auc: {auc}")

                if f1_score > best_f1_score:
                    save_new_model_and_delete_last(
                        model, os.path.join(output_dir, f"best_model_{f1_score:.4f}.pt"), delete_symbol="best_model")
                    best_f1_score = f1_score
                    logging.info(f"best f1:{f1_score}, acc: {acc}, auc: {auc}")
                if (step + 1) % save_steps == 0:
                    torch.save(model.state_dict(), os.path.join(output_dir, f"tmp_model_ep{step}_{f1_score:.4f}.pt"))

        process_bar.update(1)

save_new_model_and_delete_last(
    model, os.path.join(output_dir, f"final_model_{f1_score:.4f}.pt"), delete_symbol="final_model")