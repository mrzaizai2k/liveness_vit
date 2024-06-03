import sys
sys.path.append("")
import cv2
import pandas as pd
import os
import numpy as np
from src.Utils.utils import *
import tqdm

from src.face_detector import FaceDetection

# Load CSV file
csv_path = 'data/zalo/public_test_2/public_test_2.csv'
df = pd.read_csv(csv_path)

# Directories
video_root = 'data/zalo/public_test_2/public_test_2/videos'
output_root = 'data/zalo/zalo_public_test_2'
fake_dir = os.path.join(output_root, 'fake')
real_dir = os.path.join(output_root, 'real')

# Create directories if not exist
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

face_detector = FaceDetection.create(backend='yolo', model_config_path="config/face_detection.yml")


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
            face_locations, frame = face_detector.predict(frame)
            if face_locations:
                for startX, startY, endX, endY in face_locations:
                    refined =  refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
                    startX, startY, endX, endY = refined[:4].astype(int)
                    face = frame[startY:endY, startX:endX]

                    if face.size > 0:
                        save_path = os.path.join(save_dir, f"{label}_{saved_count}.jpg")
                        cv2.imwrite(save_path, face)
                        saved_count += 1

        frame_count += 1

    cap.release()


for idx, row in df.iterrows():
    # if idx == 5:
    #     break  # Remove this if you want to process all videos

    fname = row['fname']
    print("Processing video:", fname)
    if row['liveness_score'] >= 0.8:
        label = 'real' 
    elif row['liveness_score'] <= 0.2:
        label = 'fake'
    else: 
        continue

    video_path = os.path.join(video_root, fname)
    save_dir_base = real_dir if label == 'real' else fake_dir

    # Create a directory for each video inside the corresponding label directory
    save_dir = os.path.join(save_dir_base, f'zalo_{os.path.splitext(fname)[0]}')
    os.makedirs(save_dir, exist_ok=True)

    save_faces_from_video(video_path, label, face_detector, save_dir)
    

    if idx % 10 == 0:
        print("Processed videos:", idx)

print("Processing complete.")
