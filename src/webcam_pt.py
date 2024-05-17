import sys
sys.path.append("")
import os 
import numpy as np
import time
import imutils
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode
import cv2
import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import imutils

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
img_height = 112

transform_original = transforms.Compose([
    transforms.Resize(img_height),
    transforms.CenterCrop(img_height),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def refine(boxes, max_width, max_height, shift=0.1):
    """
    Refine the face boxes to suit the face landmark detection's needs.

    Args:
        boxes: [[x1, y1, x2, y2], ...]
        max_width: Value larger than this will be clipped.
        max_height: Value larger than this will be clipped.
        shift (float, optional): How much to shift the face box down. Defaults to 0.1.

    Returns:
       Refined results.
    """
    boxes = np.asarray(boxes, dtype=np.float64)

    refined = boxes.copy()
    width = refined[:, 2] - refined[:, 0]
    height = refined[:, 3] - refined[:, 1]

    # Move the boxes in Y direction
    shift = height * shift
    refined[:, 1] += shift
    refined[:, 3] += shift
    center_x = (refined[:, 0] + refined[:, 2]) / 2
    center_y = (refined[:, 1] + refined[:, 3]) / 2

    # Make the boxes squares
    square_sizes = np.maximum(width, height)
    refined[:, 0] = center_x - square_sizes / 2
    refined[:, 1] = center_y - square_sizes / 2
    refined[:, 2] = center_x + square_sizes / 2
    refined[:, 3] = center_y + square_sizes / 2

    # Clip the boxes for safety
    refined[:, 0] = np.clip(refined[:, 0], 0, max_width)
    refined[:, 1] = np.clip(refined[:, 1], 0, max_height)
    refined[:, 2] = np.clip(refined[:, 2], 0, max_width)
    refined[:, 3] = np.clip(refined[:, 3], 0, max_height)

    return refined

# Load the liveness detection model
TF_MODEL_FILE_PATH = 'weights/resnet50_112_new_7.pth' 

model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load(TF_MODEL_FILE_PATH))

model.eval()


def predict_liveness(model, face_image):
    """Predicts whether the face image is live or not using the given model.

    Args:
        model (torch.nn.Module): The loaded liveness detection model.
        face_image (np.ndarray): The face image to predict on.

    Returns:
        bool: True if the face is predicted to be live, False otherwise.
    """
    with torch.no_grad():
        prediction = model(face_image)
        _, predicted_class = torch.max(prediction, 1)

    # Interpret prediction based on your model's output classes (adapt accordingly)
    is_live = predicted_class[0] == 0  # Assuming class 0 represents "live"

    return is_live

# Load the liveness detection model
model = model

class_names = ['fake', 'real']

# Initialize the video capture
cap = cv2.VideoCapture("image_test/fake2.mp4", cv2.CAP_MSMF)
cap.set(3, 640)
cap.set(4, 480)

while True:
    try:
        ret, frame = cap.read()

        # Resize the frame to have a maximum width of 640 pixels
        frame = imutils.resize(frame, height=480, width=640)

        # Grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

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
                refined = refine([[startX, startY, endX, endY]], max_height=480, max_width=640, shift=0)[0]
                startX, startY, endX, endY = refined[:4].astype(int)

                # Extract the face ROI and preprocess it
                face = frame[startY:endY, startX:endX]
                cv2.imshow("Face", face)

                
                # face = cv2.resize(face, (32, 32))
                with torch.no_grad():
                    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    input_tensor = transform_original(face).unsqueeze(0).to(device)  # unsqueeze single image into batch of 1
                    output = model(input_tensor)
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
