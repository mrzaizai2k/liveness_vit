import sys
sys.path.append("")

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time
from sklearn.metrics import f1_score
import timm
from timm.models.layers import trunc_normal_
from ellzaf_ml.tools import EarlyStopping
from  torchvision.transforms import InterpolationMode 

from src.Utils.utils import *
from src.Utils.train import *
from src.dataset import CustomDataset

vit_config_path = 'config/vit_model.yaml'
vit_config = read_config(path = vit_config_path)
EPOCHS = vit_config['EPOCHS']
PATIENCE = vit_config['PATIENCE']
BATCH_SIZE = vit_config['BATCH_SIZE']
learning_rate = vit_config['learning_rate']
IMG_SIZE = vit_config['IMG_SIZE']
MODEL_DIR = vit_config['MODEL_DIR']
RANDOM_SEED = vit_config['RANDOM_SEED']

train_dir = vit_config['train_dir']
val_dir = vit_config['val_dir']
test_dir = vit_config['test_dir']

EPOCH_LEN = len(str(EPOCHS))
torch.manual_seed(RANDOM_SEED)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

device = torch.device(device)


transform_original = transforms.Compose([
    transforms.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_flipped = transforms.Compose([
    transforms.RandomRotation(degrees= 20),
    transforms.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

spoof_transforms = transforms.Compose([
    transforms.RandomRotation(degrees= 20),
    transforms.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(p=1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

])


train_orig = datasets.ImageFolder(train_dir, transform=transform_original)
train_flip = CustomDataset(root=train_dir,
                           general_augment_transform=transform_flipped,
                           special_augment_transform=spoof_transforms,
                           special_classes=['fake']) # check this

train_data_combined = ConcatDataset([train_orig, train_flip])
train_loader = DataLoader(train_data_combined, batch_size=BATCH_SIZE, shuffle=True)

val_orig = datasets.ImageFolder(val_dir, transform=transform_original)
val_flip = CustomDataset(root=val_dir,
                           general_augment_transform=transform_flipped,
                           special_augment_transform=spoof_transforms,
                           special_classes=['fake'])

val_data_combined = ConcatDataset([val_orig, val_flip])
val_loader = DataLoader(val_data_combined, batch_size=BATCH_SIZE, shuffle=False)

test_orig = datasets.ImageFolder(test_dir, transform=transform_original)
test_loader = DataLoader(test_orig, batch_size=BATCH_SIZE, shuffle=False)


print(len(train_data_combined),len(val_data_combined),len(test_orig))

def train_one_epoch(model, train_loader, device, optimizer, criterion, scheduler_lr=None):
    model.train()
    total_loss, total_f1, total_correct, total_samples, total_grad = 0, 0, 0, 0, 0

    for idx, (train_x_data, train_y_data) in enumerate(train_loader):
        train_x_data, train_y_data = train_x_data.to(device), train_y_data.to(device)
        optimizer.zero_grad()
        y_pred = model(train_x_data)

        loss = criterion(y_pred, train_y_data)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        _, y_pred_max = torch.max(y_pred, dim=1)
        f1_score_val = f1_score(train_y_data.cpu(), y_pred_max.cpu(), average="macro")
        total_correct += (y_pred_max == train_y_data).sum().item()
        total_samples += train_y_data.size(0)
        total_loss += loss.item()
        total_f1 += f1_score_val.item()
        total_grad += grad_norm

        if idx % 10 == 0 and idx > 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f'[{e:>{EPOCH_LEN}}/{EPOCHS:>{EPOCH_LEN}}] [{idx}/{len(train_loader)}] '
                f'lr {lr:.10f} | loss {loss:.10f} ({total_loss/idx:.4f}) | '
                f'grad_norm {grad_norm:.4f} ({total_grad/idx:.4f}) | '
            )

    if scheduler_lr:
        scheduler_lr.step()
    avg_loss = total_loss / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)
    avg_grad = total_grad / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, avg_f1, accuracy, avg_grad


model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)
trunc_normal_(model.head.weight, mean=0.0, std=0.02)
model = model.to(device)
model.load_state_dict(
    torch.load(
        "models/liveness/weights/vit_teacher_one.pth"
    )
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler_lr = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])

early_stopping = EarlyStopping(
                    patience=PATIENCE,
                    verbose=True,
                    path=MODEL_DIR,
                )

start_time = time.time()
for e in range(1, EPOCHS + 1):
    avg_loss_train, avg_f1_train, avg_accuracy_train, avg_grad = train_one_epoch(model, train_loader, device,
                                                                           optimizer, criterion, scheduler_lr)
    avg_loss_val, avg_f1_val, avg_accuracy_val = validate_one_epoch(model, val_loader, device, criterion)

    time_taken = time.time() - start_time
    time_format = time.strftime("%H:%M:%S", time.gmtime(time_taken))

    print(
    f"[{e:>{EPOCH_LEN}}/{EPOCHS:>{EPOCH_LEN}}] Loss: {avg_loss_train:.5f} | "
    + f"F1-score: {avg_f1_train:.3f} | Acc: {avg_accuracy_train:.3f} | "
    + f"Val Loss: {avg_loss_val:.3f} | Val F1: {avg_f1_val:.3f} | "
    + f"Val Acc: {avg_accuracy_val:.3f} | {time_format}s | "
    + f"Grad: {avg_grad:.5f}"
    )


    early_stopping(avg_loss_val, model)
    if early_stopping.early_stop:
        print(f"Early stopping after {e} Epochs")
        break

model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(
    torch.load(
        MODEL_DIR
    )
)
model.eval()

evaluate_model(model, test_loader, device=device)