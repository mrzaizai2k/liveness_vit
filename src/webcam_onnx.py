import sys
sys.path.append("")

import cv2
import time
from src.Utils.utils import *
from src.face_detector import FaceDetection
from src.model import VisionTransformerModel, ResnetModel


# Load the Caffe model for face detection
face_detector = FaceDetection.create(backend='yolo', model_config_path="config/face_detection.yml")
liveness_model = VisionTransformerModel(model_config_path="config/vit_inference.yml")

# Initialize webcam
cap = cv2.VideoCapture("image_test/vid.mp4")
  
while True:
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

            pred_class , prob = liveness_model.predict(face)

            draw_image(frame, pred_class=pred_class, prob=prob, location=[startX, startY, endX, endY])
            
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


# Release the capture
cap.release()
cv2.destroyAllWindows()