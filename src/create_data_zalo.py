import sys
sys.path.append("")
import cv2
import pandas as pd
import os
import numpy as np
from src.Utils.utils import *

# Load CSV file
csv_path = 'data/zalo/train/train/label.csv'
df = pd.read_csv(csv_path)

# Directories
video_root = 'data/zalo/train/train/videos/'
output_root = 'data/zalo/new_zalo'
fake_dir = os.path.join(output_root, 'fake')
real_dir = os.path.join(output_root, 'real')

# Create directories if not exist
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

# Load Caffe model
caffe_model = "models/face_detector/deploy.prototxt"
caffe_weights = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(caffe_model, caffe_weights)

def save_faces_from_video(video_path, label, net, save_dir, frames_per_second=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frames_per_second
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.96:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Extract and refine the face box
                    if abs(endX - startX) < 50 or abs(endY - startY) < 50:
                        continue
                    
                    # start_time = time.time()
                    refined = refine([[startX, startY, endX, endY]], max_height=h, max_width=w, shift=0)[0]
                    startX, startY, endX, endY = refined[:4].astype(int)

                    # Extract the face ROI and preprocess it
                    face = frame[startY:endY, startX:endX]

                    if face.size > 0:
                        save_path = os.path.join(save_dir, f"{label}_{frame_count}_{saved_count}.jpg")
                        cv2.imwrite(save_path, face)
                        saved_count += 1

        frame_count += 1

    cap.release()

for idx, row in df.iterrows():
    fname = row['fname']
    print("fname", fname)
    label = 'real' if row['liveness_score'] == 1 else 'fake'
    video_path = os.path.join(video_root, fname)
    save_dir = real_dir if label == 'real' else fake_dir

    save_faces_from_video(video_path, label, net, save_dir)
    # if idx==3:
    #     break 
    if idx %10==0:
        print("idx",idx)

print("Processing complete.")
