import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from typing import Literal


class FaceDetection:
    def __init__(self, model_config=None, model_config_path=None):
        # Load configuration (assuming `_read_config` is defined elsewhere)
        self.model_config = self._read_config(path=model_config_path) if model_config_path else model_config

        # Common parameters from config
        self.min_face_size = self.model_config.get("min_face_size")
        self.max_face_count = self.model_config.get("max_face_count")
        self.face_threshold = self.model_config.get("face_threshold")

    def _read_config(self, path='config/config.yaml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def _resize_frame(self, frame):
        (h, w) = frame.shape[:2]

        if w > h:
            new_frame = cv2.resize(frame, (640,480))
        else:
            new_frame = cv2.resize(frame, (300,400))

        (new_height, new_width) = new_frame.shape[:2]

        return new_height, new_width, new_frame
    
    @staticmethod
    def create(backend:Literal['opencv', 'yolo']= 'yolo', model_config=None, model_config_path=None):
        if backend == 'opencv':
            return FaceOpenCV(model_config, model_config_path)
        elif backend == 'yolo':
            return FaceYolo(model_config, model_config_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        


    def check_valid_face(self, frame, face_location:list, confidence:float):
        """
        Filters detected faces based on confidence threshold and minimum size.
        """
        [startX, startY, endX, endY]= face_location 
        height, width = frame.shape[:2]
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(width, endX)
        endY = min(height, endY)

        # filter out weak detections
        if confidence <= self.face_threshold:
            return False, None 

        if startX > width  or  startY > height:
            return False, None 

        if (abs(endX - startX) <= self.min_face_size) or (abs(endY-startY) <= self.min_face_size):
            return False, None

        face_location = [startX, startY, endX, endY]
        return True, face_location


class FaceOpenCV(FaceDetection):
    def __init__(self, model_config=None, model_config_path=None):
        super().__init__(model_config, model_config_path)
        self.caffe_model = self.model_config.get("caffe_model")
        self.caffe_weights = self.model_config.get("caffe_weights")
        self.face_model = cv2.dnn.readNetFromCaffe(self.caffe_model, self.caffe_weights)

    def detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.face_model.setInput(blob)
        detections = self.face_model.forward()    
        return detections
    
    def predict(self, frame):
        face_locations = []

        h,w,frame = self._resize_frame(frame)
        
        detections = self.detect_faces(frame)

        for i in range(0, min(detections.shape[2], self.max_face_count)): # process 5 largest faces

            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            is_valid_face, face_location = self.check_valid_face(frame, face_location =[startX, startY, endX, endY], confidence=confidence)
            if not is_valid_face:
                continue

            face_locations.append(face_location)

        return face_locations, frame



class FaceYolo(FaceDetection):
    def __init__(self, model_config=None, model_config_path=None):
        super().__init__(model_config, model_config_path)
        self.face_model_path = self.model_config.get("yolo_model")
        self.face_model = YOLO(self.face_model_path)  # Load YOLO model

    
    def detect_faces(self, frame, yolo_model):
        '''Return image with phone locations are black'''
        results = yolo_model(frame,verbose=False)
        face_boxes = results[0].boxes  # Boxes object for bbox outputs
        return face_boxes
        

    def predict(self, frame):
        face_locations = []

        h,w,frame = self._resize_frame(frame)


        face_boxes = self.detect_faces(frame, yolo_model=self.face_model)
        if face_boxes:
            for box in face_boxes:  
                confidence = box.conf.cpu().numpy()[0]

                startX, startY, endX, endY = box.xyxy.cpu().numpy()[0].astype("int")

                is_valid_face, face_location = self.check_valid_face(frame, face_location =[startX, startY, endX, endY], confidence=confidence)
                if not is_valid_face:
                    continue

                face_locations.append(face_location)

        
        return face_locations, frame
            
