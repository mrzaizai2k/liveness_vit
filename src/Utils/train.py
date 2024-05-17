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
import torchvision
from torchvision import datasets, models, transforms
from ellzaf_ml.tools import EarlyStopping


def validate_one_epoch(model, val_loader, device, criterion, scheduler_lr=None):
    model.eval()
    total_loss, total_f1, total_correct, total_samples = 0, 0, 0, 0

    with torch.no_grad():
        for val_x_data, val_y_data in val_loader:
            val_x_data, val_y_data = val_x_data.to(device), val_y_data.to(device)
            y_val_pred = model(val_x_data)
            val_loss = criterion(y_val_pred, val_y_data)

            _, y_val_pred_max = torch.max(y_val_pred, dim=1)
            f1_score_val = f1_score(val_y_data.cpu(), y_val_pred_max.cpu(), average="macro")
            total_correct += (y_val_pred_max == val_y_data).sum().item()
            total_samples += val_y_data.size(0)
            total_loss += val_loss.item()
            total_f1 += f1_score_val.item()

    avg_loss = total_loss / len(val_loader)
    if scheduler_lr:
        scheduler_lr.step(avg_loss)
    avg_f1 = total_f1 / len(val_loader)
    accuracy = total_correct / total_samples
    return avg_loss, avg_f1, accuracy


def evaluate_model(model, test_loader, device='cpu'):

    with torch.no_grad():
        correct = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            tp += (pred.eq(1) & target.eq(1).view_as(pred)).sum().item()
            tn += (pred.eq(0) & target.eq(0).view_as(pred)).sum().item()
            fp += (pred.eq(1) & target.eq(0).view_as(pred)).sum().item()
            fn += (pred.eq(0) & target.eq(1).view_as(pred)).sum().item()

            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        far = fp / (fp + tn)
        frr = fn / (fn + tp)

        recall = tp / (tp + fn)

        hter = (far + frr ) / 2

        print(f"test acc: {accuracy * 100}%")
        print(f"recall: {recall * 100}%")
        print(f"far: {far * 100}%")
        print(f"frr: {frr * 100}%")
        print(f"hter: {hter * 100}%")