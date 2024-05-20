import io
import timm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Define the inference function
def predict_image(img_bytes, model, transform, device):
    img = Image.open(io.BytesIO(img_bytes))
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        return pred, prob
    
def initialize_model(model_path:str = 'models/liveness/weights/vit_teacher_one.pth', device:str = 'cpu'):
    # Load the model
    model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k')
    model.head = nn.Linear(model.head.in_features, 2)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()


    data_config = timm.data.resolve_model_data_config(model)
    print("data_config", data_config)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform