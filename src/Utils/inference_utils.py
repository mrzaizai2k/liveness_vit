import io
import timm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2

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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

def predict_onnx(ort_inputs, ort_session):
    output = ort_session.run(None, ort_inputs)
    output = np.array(output[0])  # Ensure output is a NumPy array
    # Determine the prediction
    pred_class = np.argmax(output, axis=1)
    prob = softmax(output[0])

    return pred_class, prob


def preprocess_image(image, target_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to slightly larger than the target size to maintain aspect ratio for center cropping
    height, width = image.shape[:2]
    scale = max(target_size / height, target_size / width)
    new_height, new_width = int(height * scale), int(width * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Center crop the image
    start_x = (new_width - target_size) // 2
    start_y = (new_height - target_size) // 2
    image = image[start_y:start_y + target_size, start_x:start_x + target_size]

    # Convert image to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Normalize the image
    mean = np.array(mean, dtype=np.float32).reshape((1, 1, 3))
    std = np.array(std, dtype=np.float32).reshape((1, 1, 3))
    image = (image - mean) / std

    # Convert image to tensor format (C, H, W)
    image = np.transpose(image, (2, 0, 1))

    return image
