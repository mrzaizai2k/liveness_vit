import sys
sys.path.append("")

from PIL import Image
import timm
import numpy as np
import torch
from torchvision import transforms
from src.Utils.inference_utils import *

model_path = "models/liveness/weights/vit_teacher.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k')
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

model.eval()


data_config = timm.data.resolve_model_data_config(model)
print("data_config", data_config)
transforms = timm.data.create_transform(**data_config, is_training=False)


# 1. Load the image
# img_path = 'face_test/fake/0010/0010_01_01_03_0.jpg'
img_path = 'image_test/realtwo.jpg'
img = Image.open(img_path)
img = transforms(img)
img = img.unsqueeze(0)
img = img.to(device)
print(img.shape)
# 2. Load the model with specified weights

output = model(img)
print("output ", output)
pred = output.argmax(dim=1, keepdim=True)
print("pred ", pred)
prob = torch.nn.functional.softmax(output[0], dim=0)
print("prob ", prob)


save_path = "models/liveness/weights/vit_teacher.onnx"

torch_out = model(img)

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

if torch.cuda.is_available():
    ort_session = onnxruntime.InferenceSession(save_path, providers=['CUDAExecutionProvider'])
else:
    ort_session = onnxruntime.InferenceSession(save_path, providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")