import sys
sys.path.append("")
import numpy as np
import json
import requests
import warnings

import ast
warnings.filterwarnings("ignore")

from src.Utils.utils import *


def test_liveness_bytes_api(img_path, root_url, face_detect=True):
    url = f"{root_url}/v1/faceservice/liveness/predict?face_detect={face_detect}"
    img = cv2.imread(img_path)
    _, img_buffer_arr = cv2.imencode(".jpg", img)
    img_bytes = img_buffer_arr.tobytes()
    
    headers = {'Content-Type': 'image/jpeg'}
    res = requests.post(url, data=img_bytes, headers=headers)
    print(f'result: {res.json()}')



def test_liveness_base64_api(img_path, root_url, face_detect=True):
    url = f"{root_url}/v1/faceservice/liveness/predict?face_detect={face_detect}"
    base64_img = convert_img_path_to_base64(img_path)
    res = requests.post(url, json ={"img":base64_img})
    # print(f'Status Code: {res.status_code}')
    print(f'result: {res.json()}')


def show_images(base64_images):
    """Decode and display base64 encoded images using OpenCV."""
    for i, base64_image in enumerate(base64_images):
        # Decode base64 to binary image data
        img_data = base64.b64decode(base64_image)
        
        # Convert binary data to numpy array
        np_arr = np.frombuffer(img_data, np.uint8)
        
        # Decode numpy array to OpenCV image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Display the image
        if img is not None:
            cv2.imshow(f'Image {i+1}', img)
        else:
            print(f"Failed to decode image {i+1}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    connection_config_path = "config/connection.yml"
    connection_config = read_config(path=connection_config_path)     

    img_path = 'image_test/realtwo.jpg'
   
    root_url = "http://192.168.2.77:8090" #Bao
    test_liveness_bytes_api(img_path=img_path, root_url=root_url, face_detect=True)
    test_liveness_base64_api(img_path=img_path, root_url=root_url, face_detect=True)




