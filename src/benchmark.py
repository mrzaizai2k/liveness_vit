import sys
sys.path.append("")

import torch
from torchvision import datasets


import timm
from  torchvision.transforms import InterpolationMode 
from torchvision.transforms import v2

from src.Utils.utils import *
from src.Utils.train import *



vit_config_path = 'config/vit_model.yaml'
vit_config = read_config(path = vit_config_path)

BATCH_SIZE = vit_config['BATCH_SIZE']
IMG_SIZE = vit_config['IMG_SIZE']
RANDOM_SEED = vit_config['RANDOM_SEED']
NUM_CLASSES = vit_config['NUM_CLASSES']

MODEL_DIR = vit_config['MODEL_DIR']
MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='vit')
vit_config['MODEL_DIR'] = MODEL_DIR

test_dir = vit_config['test_dir']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform_original = v2.Compose([
    v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_orig = datasets.ImageFolder(test_dir, transform=transform_original)
test_loader = DataLoader(test_orig, batch_size=BATCH_SIZE, shuffle=False)


model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k')
model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
model = model.to(device)
model.load_state_dict(
    torch.load(
        "models/liveness/vit_teacher_4_may_3.pth"
    )
)
model.eval()

evaluate_model(model, test_loader, device=device, threshold=0.5)