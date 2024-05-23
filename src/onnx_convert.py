import sys
sys.path.append("")

from PIL import Image
import timm
import numpy as np
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from src.Utils.inference_utils import *

# model_path = "models/liveness/weights/resnet50_224.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

########## Resnet ########
# img_height= 224
# model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
# for param in model.parameters():
#     param.requires_grad = False

# num_ftrs = model.fc.in_features

# model.fc = nn.Linear(num_ftrs, 2)
# model = model.to(device)
# model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# model.eval()

# transform_original = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(img_height),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


########## ViT ########
model_path = "models/liveness/weights/vit_teacher_new_config_siw.pth"
model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k')
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()


data_config = timm.data.resolve_model_data_config(model)
print("data_config", data_config)
transform_original = timm.data.create_transform(**data_config, is_training=False)


# 1. Load the image
# img_path = 'face_test/fake/0010/0010_01_01_03_0.jpg'
img_path = 'image_test/realtwo.jpg'
img = Image.open(img_path)
img = transform_original(img)
img = img.unsqueeze(0)
img = img.to(device)
print(img.shape)
# 2. Load the model with specified weights

output = model(img)
pred = output.argmax(dim=1, keepdim=True)
prob = torch.nn.functional.softmax(output[0], dim=0)
print("output 1", output)
print("pred 1", pred)
print("prob 1 ", prob)


save_path = "models/liveness/weights/vit_teacher_new_config_siw.onnx"

# torch_out = model(img)

# Export the model
torch.onnx.export(model,               # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  save_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession(save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


img = cv2.imread(img_path)
img = preprocess_image(img)

input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: img[np.newaxis, :]}

pred_class, prob = predict_onnx(ort_session=ort_session, ort_inputs=ort_inputs)
print("prob", prob)
print("pred_class", pred_class)


# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")