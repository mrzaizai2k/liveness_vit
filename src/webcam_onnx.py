import sys
sys.path.append("")

import cv2
import numpy as np
import onnxruntime

from src.Utils.utils import *
from src.Utils.inference_utils import *

model_dir = "models/liveness/weights/vit_teacher_one.onnx"
img_height = 224

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load the Caffe model for face detection
caffe_model = "models/face_detector/deploy.prototxt"
caffe_weights = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(caffe_model, caffe_weights)

ort_session = onnxruntime.InferenceSession(model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


# Initialize webcam
cap = cv2.VideoCapture("image_test/vid.mp4", cv2.CAP_MSMF)

while True:
    try: 
        ret, frame = cap.read()

        if not ret:
            continue

        # frame = imutils.resize(frame, height=480, width=640)
        frame = cv2.resize(frame, (640,480))

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        if not detections.shape[2] > 0:
            continue 

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.8:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                if abs(endX - startX) < 100 or abs(endY-startY) < 100:
                    continue

                start_time = time.time()

                refined =  refine([[startX, startY, endX, endY]], max_height=480, max_width=640)[0]

                startX, startY, endX, endY = refined[:4].astype(int)

                face = frame[startY:endY, startX:endX]
                face = preprocess_image(face)
                input_name = ort_session.get_inputs()[0].name
                ort_inputs = {input_name: face[np.newaxis, :]}

                # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
                pred_class , prob = predict_onnx(ort_inputs=ort_inputs, ort_session=ort_session)
                print("prob", prob)

                if pred_class[0] == 1:
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
    except:
        print("error")

# Release the capture
cap.release()
cv2.destroyAllWindows()
