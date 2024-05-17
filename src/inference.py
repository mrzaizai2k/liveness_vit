import sys
sys.path.append("")

from PIL import Image
import timm
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform_original = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 1. Load the image
# img_path = 'face_test/fake/0010/0010_01_01_03_0.jpg'
img_path = 'image_test/fakeone.jpg'
img = Image.open(img_path)
img = transform_original(img)
img = img.unsqueeze(0)
img = img.to(device)
# 2. Load the model with specified weights


model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(
    torch.load(
        "weights/vit_teacher.pth"
    )
)
model.eval()

# 3. Get model-specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)


# # 4. Apply transforms and classify the image

# input_tensor = transforms(img).unsqueeze(0).to(device)  # unsqueeze single image into batch of 1
output = model(img)
print("output ", output)
pred = output.argmax(dim=1, keepdim=True)
print("pred ", pred)
