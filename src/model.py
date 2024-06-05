import sys
sys.path.append("")

import cv2
import numpy as np
import onnxruntime
import yaml

class LivenessModel:
    def __init__(self, model_config=None, model_config_path=None):
        """
        Initialize the Liveness Model.

        Args:
            model_config_path (str): Path to the model configuration file.

        Raises:
            ValueError: If the model directory is not specified or not an ONNX file.
        """
        if model_config_path is not None:
            self.model_config_path = model_config_path
            self.model_config = self._read_config(path=self.model_config_path)
        elif model_config is not None:
            self.model_config = model_config

        self.model_dir = self.model_config.get("MODEL_DIR")
        self._check_model_onnx()
        self.target_resize = self.model_config.get("target_resize")
        self.target_crop = self.model_config.get("target_crop")
        self.mean = self.model_config.get("mean")
        self.std = self.model_config.get("std")
        self.initiate_model()

    def _read_config(self, path='config/config.yaml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def _check_model_onnx(self):
        """
        Check if the model directory is specified and points to an ONNX file.
        Raises:
            ValueError: If the model directory is not specified or not an ONNX file.
        """
        # Check if the model directory is specified and not None
        if self.model_dir is None:
            raise ValueError("Model directory not specified in configuration.")

        # Check the file extension to determine if it's an ONNX model
        if not self.model_dir.endswith('.onnx'):
            raise ValueError("You should convert model to onnx first.")

    def initiate_model(self):
        """
        Initialize the ONNX runtime inference session.
        """
        self.ort_session = onnxruntime.InferenceSession(self.model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name

    def _calculate_softmax(self, x):
        """
        Calculate the softmax of the input array.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
        return e_x / e_x.sum(axis=0)

    def _predict_onnx(self, ort_inputs):
        """
        Perform inference using the ONNX model.

        Args:
            ort_inputs (dict): Dictionary of input data for ONNX model.

        Returns:
            tuple: Tuple containing predicted class and probabilities of each class.
        """
        output = self.ort_session.run(None, ort_inputs)
        output = np.array(output[0])  # Ensure output is a NumPy array
        # Determine the prediction
        pred_class = np.argmax(output, axis=1)
        prob = self._calculate_softmax(output[0])

        return pred_class, prob
        
    def predict(self, image):
        """
        Predict the class and probability for the input image.

        Args:
            image (np.ndarray): Input image in BGR format. The image should be 224x224 image

        Returns:
            tuple: Tuple containing predicted class and probabilities.
        """
        image = self.preprocess(image)
        try:
            ort_inputs = {self.input_name: image[np.newaxis, :]}
            pred_class , prob = self._predict_onnx(ort_inputs=ort_inputs)
        except:
            ort_inputs = {self.input_name: image.astype(np.float16)[np.newaxis, :]}
            pred_class , prob = self._predict_onnx(ort_inputs=ort_inputs)
        return pred_class , prob 

class VisionTransformerModel(LivenessModel):
    def __init__(self, model_config=None, model_config_path=None):
        super().__init__(model_config, model_config_path)

    def preprocess(self, image):
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image in tensor format (C, H, W).
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to slightly larger than the target size to maintain aspect ratio for center cropping
        height, width = image.shape[:2]
        scale = self.target_resize / min(height, width)
        new_height, new_width = int(np.ceil(height *  scale)), int(np.ceil(width * scale))

        # print('new_height, new_width', new_height, new_width)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA )

        # Center crop the image
        start_x = (new_width - self.target_crop) // 2
        start_y = (new_height - self.target_crop) // 2
        image = image[start_y:start_y + self.target_crop, start_x:start_x + self.target_crop]
        # print("crop image.shape", image.shape)
        # print("start_x", start_x)
        # print("start_y", start_y)


        # Convert image to float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize the image
        mean = np.array(self.mean, dtype=np.float32).reshape((1, 1, 3))
        std = np.array(self.std, dtype=np.float32).reshape((1, 1, 3))
        image = (image - mean) / std
        # print("image.shape[0]", image.shape[0])
        # print("image.shape", image.shape)
        # Convert image to tensor format (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image

class ResnetModel(LivenessModel):
    def __init__(self, model_config=None, model_config_path=None):
        super().__init__(model_config, model_config_path)

    def preprocess(self, image):
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image in tensor format (C, H, W).
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to slightly larger than the target size to maintain aspect ratio for center cropping
        height, width = image.shape[:2]

        scale = self.target_resize / min(height, width)
        new_height, new_width = int(np.ceil(height  * scale)), int(np.ceil(width * scale))
        image = cv2.resize(image, (new_width, new_height))

        # Center crop the image
        start_x = (self.target_resize - self.target_crop) // 2
        start_y = (self.target_resize - self.target_crop) // 2
        image = image[start_y:start_y + self.target_crop, start_x:start_x + self.target_crop]

        # Convert image to float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize the image
        mean = np.array(self.mean, dtype=np.float32).reshape((1, 1, 3))
        std = np.array(self.std, dtype=np.float32).reshape((1, 1, 3))
        image = (image - mean) / std

        # Convert image to tensor format (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image


    
if __name__ == "__main__":
    img_path = 'image_test/real (6).jpg'
    img = cv2.imread(img_path)
    model = VisionTransformerModel(model_config_path="config/vit_inference.yml")
    pred_class , prob = model.predict(img)
    print("pred_class", pred_class)
    print("prob", prob)
    score = round(prob[pred_class[0]].item(),3)

    if pred_class[0] == 1:
            data = {"status_code": 200,
                    "liveness": 'real',
                    "score": score}
    else:
        data = {"status_code": 200,
                "liveness": 'fake',
                "score": score}
    print('data', data)
    print("prob 1", prob[1])


    model = ResnetModel(model_config_path="config/resnet_inference.yml")
    pred_class , prob = model.predict(img)
    print("pred_class", pred_class)
    print("prob", prob)
    score = round(prob[pred_class[0]].item(),3)

    if pred_class[0] == 1:
            data = {"status_code": 200,
                    "liveness": 'real',
                    "score": score}
    else:
        data = {"status_code": 200,
                "liveness": 'fake',
                "score": score}
    print("prob 1", prob[1])
    
    print('data', data)
