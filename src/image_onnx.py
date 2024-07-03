
import sys
sys.path.append("")

import cv2
import numpy as np
import onnxruntime

from src.Utils.utils import *
from src.Utils.inference_utils import *
from src.face_detector import FaceDetection
from src.model import VisionTransformerModel, ResnetModel


# Load the Caffe model for face detection
face_detector = FaceDetection.create(backend='opencv', model_config_path="config/face_detection.yml")
liveness_model = VisionTransformerModel(model_config_path="config/vit_inference.yml")

# Initialize webcam
# frame = base64_path_to_image(base64_path="image_test/fakeios.txt")
frame = cv2.imread("image_test/ong-hoan-ptgd-nab-3933.jpg")
print('frame.shape', frame.shape)

face_locations, frame = face_detector.predict(frame)

# loop over the detections
for startX, startY, endX, endY in face_locations:
    refined =  refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
    startX, startY, endX, endY = refined[:4].astype(int)
    face = frame[startY:endY, startX:endX]
    pred_class , prob = liveness_model.predict(face)

    draw_image(frame, pred_class=pred_class, prob=prob, location=[startX, startY, endX, endY])

        
# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)

cv2.destroyAllWindows()
