import cv2
import numpy as np
from PIL import Image
import timm
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load the Caffe model for face detection
caffe_model = "models/face_detector/deploy.prototxt"
caffe_weights = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(caffe_model, caffe_weights)

# Load the ViT model with specified weights
model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load("weights/vit_teacher_224.pth", map_location=torch.device(device)))
model.eval()

# Get model-specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

def refine(boxes, max_width, max_height, shift=0.1):
    """Refine the face boxes to suit the face landmark detection's needs.

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

transform_original = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize webcam
cap = cv2.VideoCapture("image_test/fake2.mp4")

while True:
    try: 
        ret, frame = cap.read()
        if not ret:
            break

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

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
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
                if abs(endX - startX) < 100 or abs(endY-startY)< 100:
                    continue
                refined =  refine([[startX, startY, endX, endY]], max_height=480, max_width=640)[0]

                startX, startY, endX, endY = refined[:4].astype(int)
                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
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


        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("error")

# Release the capture
cap.release()
cv2.destroyAllWindows()
