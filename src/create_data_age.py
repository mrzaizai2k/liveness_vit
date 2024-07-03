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
input_folder = "data/age_data"
output_folder = "data/age_data/real"

# Create output folders
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for subdir, dirs, files in os.walk(input_folder):
    subdir = subdir.replace("\\", "/")
    label = subdir.split("/")[-1].lower()  # Get label from directory name
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(subdir, file).replace("\\", "/")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            print(f'Processing {image_path}')
            
            face_locations, frame = face_detector.predict(image)
                
            for startX, startY, endX, endY in face_locations:
                refined = refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
                startX, startY, endX, endY = refined[:4].astype(int)
                face = frame[startY:endY, startX:endX]
                
                output_subfolder = output_folder
                os.makedirs(output_subfolder, exist_ok=True)
                
                output_path = os.path.join(output_subfolder, f"{label}_{os.path.splitext(file)[0]}_face.jpg").replace("\\", "/")
                
                cv2.imwrite(output_path, face)

print("Processing completed.")
