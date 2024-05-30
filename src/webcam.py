import sys
sys.path.append("")


import cv2
import numpy as np
from PIL import Image
import timm
import torch
from torchvision import  transforms
from src.face_detector import FaceDetection
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from src.Utils.utils import *
from src.Utils.inference_utils import *



model_dir = "models/liveness/weights/vit_teacher_new_config_siw.pth"
img_height= 224

map_size = int(np.sqrt(196))

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

face_detector = FaceDetection.create(backend='yolo', model_config_path="config/face_detection.yml")


# Load the ViT model with specified weights
model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k')
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load(model_dir, map_location=torch.device(device)))
# model.half()
model.eval()


data_config = timm.data.resolve_model_data_config(model)
print("data_config", data_config)
transform_original = timm.data.create_transform(**data_config, is_training=False)

# Initialize webcam
cap = cv2.VideoCapture("image_test/video1.mp4")

while True:
    # try: 
    face_locations =[] 
    ret, frame = cap.read()

    if not ret:
        continue        

    face_locations, frame = face_detector.predict(frame)
    if face_locations:
        for startX, startY, endX, endY in face_locations:

            start_time = time.time()

            refined =  refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
            startX, startY, endX, endY = refined[:4].astype(int)
            face = frame[startY:endY, startX:endX]
            # cv2.imshow("Face", face)

            
            # face = cv2.resize(face, (32, 32))
            with torch.no_grad():
                face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                input_tensor = transform_original(face).unsqueeze(0).to(device)  # unsqueeze single image into batch of 1

                model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)

                output = model(input_tensor)

                attn_map = model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach().cpu()

                cls_attn_map = model.blocks[-1].attn.cls_attn_map.mean(dim=1)
                padding = (0, 1)  # (left, right) padding for the last dimension
                cls_attn_map_padded = F.pad(cls_attn_map, padding, mode='constant', value=0) # should pad so the sqrt(map_size) = input size

                cls_weight = cls_attn_map_padded.view(map_size,map_size).detach().cpu()

                img_resized = transform_original(face).permute(1, 2, 0).cpu() * 0.5 + 0.5
                cls_resized = F.interpolate(cls_weight.view(1, 1, map_size,map_size), (224, 224), mode='bicubic').view(224, 224, 1).cpu()

                show_img2_vid(img_resized, cls_resized)

                print("output ", output)
                pred = output.argmax(dim=1, keepdim=True)
                print("pred ", pred)
                if pred[0][0] == 1:
                    text = 'real'
                    color = (0,255,0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                else:
                    text = 'fake'
                    color = (255,0,0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            end_time = time.time()

            # Calculate FPS
            processing_time = end_time - start_time
            fps = 1 / processing_time if processing_time > 0 else 0
            
            # Draw FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # except:
    #     print("error")

# Release the capture
cap.release()
cv2.destroyAllWindows()
