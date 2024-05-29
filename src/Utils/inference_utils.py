import io
import timm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_img2(img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

def show_img2_vid(img1, img2, alpha=0.2):
    # Ensure img1 is 3-channel and img2 is 1-channel
    img1 = np.asarray(img1)
    img1 = img1[:,:,1]
    img2 = np.asarray(img2)


    # Normalize img2 to [0, 1] range and scale it according to alpha
    img2_normalized = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    img2_scaled = img2_normalized * (3 * alpha)

    # Overlay img2 on img1 with alpha blending
    blended_img = cv2.addWeighted(img1, alpha, img2_scaled, 1-alpha, 0)
    

    # combined = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv2.imshow('attention map', blended_img)
    # return combined


def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

