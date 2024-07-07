import sys
sys.path.append("")

import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import wandb

from sklearn.metrics import f1_score
import timm
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from ellzaf_ml.tools import EarlyStopping
from  torchvision.transforms import InterpolationMode 
from torchvision.transforms import v2
from torch.utils.data import default_collate

from src.Utils.utils import *
from src.Utils.train import *
from src.Utils.augment import *
from src.dataset import CustomDataset

vit_config_path = 'config/vit_model.yaml'
vit_config = read_config(path = vit_config_path)

MODEL_TYPE = vit_config['MODEL_TYPE']
EPOCHS = vit_config['EPOCHS']
PATIENCE = vit_config['PATIENCE']
BATCH_SIZE = vit_config['BATCH_SIZE']
learning_rate = vit_config['learning_rate']
IMG_SIZE = vit_config['IMG_SIZE']
RANDOM_SEED = vit_config['RANDOM_SEED']
NUM_CLASSES = vit_config['NUM_CLASSES']
WEIGHT_DECAY = vit_config['WEIGHT_DECAY']
LR_WARMUP = vit_config['LR_WARMUP']
CLIP_GRAD_NORM = vit_config['CLIP_GRAD_NORM']

MODEL_DIR = vit_config['MODEL_DIR']
MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='nextvit')
vit_config['MODEL_DIR'] = MODEL_DIR


train_dir = vit_config['train_dir']
val_dir = vit_config['val_dir']
test_dir = vit_config['test_dir']

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="liveness_vit",
    name= os.path.splitext(os.path.basename(MODEL_DIR))[0],
    notes=vit_config['notes'],
    tags=vit_config['tags'],
    # track hyperparameters and run metadata
    config=vit_config
)


EPOCH_LEN = len(str(EPOCHS))
torch.manual_seed(RANDOM_SEED)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# cutmix = v2.CutMix(num_classes=NUM_CLASSES)
# mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=0.2)
# cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[0.8, 0.5])


model = timm.create_model(MODEL_TYPE, num_classes=NUM_CLASSES, pretrained=True)
# model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
# trunc_normal_(model.head.weight, mean=0.0, std=0.02)
model = model.to(device)
# model.load_state_dict(
#     torch.load(
#         "models/liveness/weights/vit_2024_06_06_4.pth"
#     )
# )

data_config = timm.data.resolve_model_data_config(model)
print("data_config", data_config)
transform_original = timm.data.create_transform(**data_config, is_training=False) #training = False as I want to keep the original

# transform_original = v2.Compose([
#     v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
#     v2.CenterCrop(IMG_SIZE),
#     v2.ToTensor(),
#     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])


transform_flipped = v2.Compose([
    v2.TrivialAugmentWide(),
    v2.RandomRotation(degrees= 50),
    v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.RandomApply([v2.RandomResizedCrop(IMG_SIZE)] , p=0.7),
    v2.RandomErasing(p=0.2),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    v2.RandomAdjustSharpness(sharpness_factor = 2,p=0.5),
    v2.RandomAdjustSharpness(sharpness_factor = 0, p=0.5),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))] , p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.JPEG((5, 50)),
    v2.ToTensor(),
    v2.RandomApply([AddSaltAndPepperNoise(salt_prob=0.02, pepper_prob=0.02)], p=0.5),
    v2.Normalize(mean=data_config['mean'], std=data_config['std']),
])

spoof_transforms = v2.Compose([
    v2.TrivialAugmentWide(),
    v2.RandomRotation(degrees= 50),
    v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.RandomApply([v2.RandomResizedCrop(IMG_SIZE)] , p=0.7),
    v2.RandomErasing(p=0.2),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    v2.RandomAdjustSharpness(sharpness_factor = 2,p=0.5),
    v2.RandomAdjustSharpness(sharpness_factor = 0, p=0.5),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))] , p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.JPEG((5, 50)),
    v2.ToTensor(),
    v2.RandomApply([AddSaltAndPepperNoise(salt_prob=0.02, pepper_prob=0.02)], p=0.5),
    v2.Normalize(mean=data_config['mean'], std=data_config['std']),
])


train_orig = datasets.ImageFolder(train_dir, transform=transform_original)
train_flip = CustomDataset(root=train_dir,
                           general_augment_transform=transform_flipped,
                           special_augment_transform=spoof_transforms,
                           special_classes=['fake']) # check this

train_data_combined = ConcatDataset([train_orig, train_flip])
train_loader = DataLoader(train_data_combined, batch_size=BATCH_SIZE, shuffle=True,)

val_orig = datasets.ImageFolder(val_dir, transform=transform_original)
val_loader = DataLoader(val_orig, batch_size=BATCH_SIZE, shuffle=False)

test_orig = datasets.ImageFolder(test_dir, transform=transform_original)
test_loader = DataLoader(test_orig, batch_size=BATCH_SIZE, shuffle=False)


print(len(train_data_combined),len(val_orig),len(test_orig))

def train_one_epoch(model, train_loader, optimizer, criterion, scheduler_lr=None, device = 'cpu'):
    model.train()
    total_loss, total_f1, total_correct, total_samples, total_grad = 0, 0, 0, 0, 0

    for idx, (train_x_data, train_y_data) in enumerate(train_loader):
        train_x_data, train_y_data = train_x_data.to(device), train_y_data.to(device)
        # train_x_data, train_y_data = cutmix_or_mixup( train_x_data, train_y_data)
        optimizer.zero_grad()
        y_pred = model(train_x_data)

        loss = criterion(y_pred, train_y_data)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        _, y_pred_max = torch.max(y_pred, dim=1)
        # f1_score_val = f1_score(train_y_data.cpu(), y_pred_max.cpu(), average="macro")
        # total_correct += (y_pred_max == train_y_data).sum().item()
        total_samples += train_y_data.size(0)
        total_loss += loss.item()
        # total_f1 += f1_score_val.item()
        total_grad += grad_norm

        if idx % 10 == 0 and idx > 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f'[{epoch:>{EPOCH_LEN}}/{EPOCHS:>{EPOCH_LEN}}] [{idx}/{len(train_loader)}] '
                f'lr {lr:.10f} | loss {loss:.10f} ({total_loss/idx:.4f}) | '
                f'grad_norm {grad_norm:.4f} ({total_grad/idx:.4f}) | '
            )

            wandb.log({
                "epoch": epoch,
                "batch": idx,
                "lr": lr,
                "loss": loss.item(),
                "avg_loss": total_loss / idx,
                "grad_norm": grad_norm,
                "avg_grad_norm": total_grad / idx
            })

    if scheduler_lr:
        scheduler_lr.step()
    avg_loss = total_loss / len(train_loader)
    # avg_f1 = total_f1 / len(train_loader)
    avg_grad = total_grad / len(train_loader)
    # accuracy = total_correct / total_samples
    return avg_loss, avg_grad #, avg_f1, accuracy, 



criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# criterion = SoftTargetCrossEntropy()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = WEIGHT_DECAY)

scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=LR_WARMUP)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler_lr = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])

early_stopping = EarlyStopping(
                    patience=PATIENCE,
                    verbose=True,
                    path=MODEL_DIR,
                )

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    avg_loss_train, avg_grad = train_one_epoch(model=model, train_loader=train_loader, device=device, 
                                                                           optimizer=optimizer, criterion=criterion, scheduler_lr=scheduler_lr)
    avg_loss_val, avg_f1_val, avg_accuracy_val = validate_one_epoch(model=model, val_loader=val_loader, device=device, criterion=criterion)

    time_taken = time.time() - start_time
    time_format = time.strftime("%H:%M:%S", time.gmtime(time_taken))

    print(
    f"[{epoch:>{EPOCH_LEN}}/{EPOCHS:>{EPOCH_LEN}}] Loss: {avg_loss_train:.5f} | "
    # + f"F1-score: {avg_f1_train:.3f} | Acc: {avg_accuracy_train:.3f} | "
    + f"Val Loss: {avg_loss_val:.3f} | Val F1: {avg_f1_val:.3f} | "
    + f"Val Acc: {avg_accuracy_val:.3f} | {time_format}s | "
    + f"Grad: {avg_grad:.5f}"
    )
    wandb.log({
        "epoch": epoch,
        "avg_loss_val": avg_loss_val,
        "avg_f1_val": avg_f1_val,
        "accuracy_val": avg_accuracy_val
    })

    early_stopping(avg_loss_val, model)
    if early_stopping.early_stop:
        print(f"Early stopping after {epoch} Epochs")
        break

model = timm.create_model(MODEL_TYPE, num_classes=NUM_CLASSES, pretrained=False)
model = model.to(device)
model.load_state_dict(
    torch.load(
        MODEL_DIR
    )
)
model.eval()

evaluate_model(model, test_loader, wandb=wandb, device=device)