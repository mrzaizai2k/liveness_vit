import sys
sys.path.append("")
import numpy as np
import cv2
import imutils


import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from src.face_detector import FaceDetection

from  torchvision.transforms import InterpolationMode 

from src.Utils.utils import *



model_dir = 'models/liveness/weights/resnet50_zalo_new_config.pth' 
img_height = 224

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

face_detector = FaceDetection.create(backend='yolo', model_config_path="config/face_detection.yml")

# Load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")


transform_original = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_height),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the liveness detection model

model = torchvision.models.resnet50()
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load(model_dir))

model.eval()


# Initialize the video capture
cap = cv2.VideoCapture("image_test/vid.mp4")

while True:
    try:
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
                    output = model(input_tensor)
                    # print("output ", output)
                    pred = output.argmax(dim=1, keepdim=True)
                    # print("pred ", pred)
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
                

        # Display the frame with the face detections
        cv2.imshow('Frame', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print('error')

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
