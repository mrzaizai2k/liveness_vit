import sys
sys.path.append("")

import os
import sys
import cv2
import numpy as np
import onnxruntime
from src.Utils.utils import *
from src.Utils.inference_utils import *
from src.face_detector import FaceDetection
from src.model import VisionTransformerModel, ResnetModel

import os
import cv2
from src.face_detector import FaceDetection
from src.model import VisionTransformerModel

# Load models
face_detector = FaceDetection.create(backend='yolo', model_config_path="config/face_detection.yml")

# Input and output directories
input_folder = "data/face_3.3G/test"
output_folder = "data/face_3.3G/test/image"

# Create output folders
os.makedirs(os.path.join(output_folder, "attack"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "real_video"), exist_ok=True)

# Function to get frame rate of a video
def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Process each video in the input folder
for subdir, dirs, files in os.walk(input_folder):
    subdir = subdir.replace("\\","/")
    for file in files:
        video_path = os.path.join(subdir, file).replace("\\","/")
        label = subdir.split("/")[-1].lower().replace("\\","/")  # Get label from directory name

        # Get frame rate of the video
        fps = get_frame_rate(video_path)
        
        # Calculate interval to save 5 frames per second
        interval = int(round(fps / 3))
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save only frames at the specified interval
            if frame_count % interval == 0:
                print(f'Processing frame {frame_count} of {video_path}')
                
                face_locations, frame = face_detector.predict(frame)
                
                for startX, startY, endX, endY in face_locations:
                    refined = refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
                    startX, startY, endX, endY = refined[:4].astype(int)
                    face = frame[startY:endY, startX:endX]
                    
                    output_subfolder = "real_video" if label == "real_video" else "attack"
                    output_path = os.path.join(output_folder, output_subfolder, f"{os.path.splitext(file)[0]}_frame{frame_count}.jpg").replace("\\","/")
                    
                    cv2.imwrite(output_path, face)
        
        cap.release()

cv2.destroyAllWindows()
