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
from  torchvision.transforms import InterpolationMode 

from src.Utils.utils import *



model_dir = 'models/liveness/weights/resnet101_224_siw.pth' 
img_height = 224

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "models/face_detector/deploy.prototxt"
modelPath = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")


transform_original = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_height),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the liveness detection model

model = torchvision.models.resnet101()
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load(model_dir))

model.eval()


# Initialize the video capture
cap = cv2.VideoCapture("image_test/vid.mp4")
cap.set(3, 640)
cap.set(4, 480)

while True:
    try:
        ret, frame = cap.read()

        if not ret:
            continue
        # Resize the frame to have a maximum width of 640 pixels
        # frame = imutils.resize(frame, height=480, width=640)
        frame = cv2.resize(frame, (640,480))


        # Grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        if not detections.shape[2] > 0:
            continue 

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.8:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the detected bounding box does not fall outside the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Extract and refine the face box
                if abs(endX - startX) < 100 or abs(endY - startY) < 100:
                    continue
                
                start_time = time.time()
                refined = refine([[startX, startY, endX, endY]], max_height=480, max_width=640, shift=0)[0]
                startX, startY, endX, endY = refined[:4].astype(int)

                # Extract the face ROI and preprocess it
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
