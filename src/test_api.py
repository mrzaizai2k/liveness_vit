import sys
sys.path.append("")
import numpy as np
import json
import requests
import warnings
import concurrent.futures
import ast

warnings.filterwarnings("ignore")

from src.Utils.utils import *
from src.Utils.logger import create_logger
logger = create_logger(logfile="logs/liveness_test.log")


def test_liveness_bytes_api(img_path, root_url, face_detect=True):
    url = f"{root_url}/v1/faceservice/liveness/predict?face_detect={face_detect}"
    img = cv2.imread(img_path)
    _, img_buffer_arr = cv2.imencode(".jpg", img)
    img_bytes = img_buffer_arr.tobytes()
    
    headers = {'Content-Type': 'image/jpeg'}
    res = requests.post(url, data=img_bytes, headers=headers)
    print(f'result: {res.json()}')
    logger.debug(msg=f"res.json: {res.json()}")



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


@timeit
def concurrent_face_recognition(img_path, root_url, face_detect, num_images=100):
    """
    Perform concurrent face recognition API calls.
    """
    pause_required = [False]  # Using a list for a mutable, shared state

    def execute_api_call(img_path, root_url):
        nonlocal pause_required
        try:
            test_liveness_bytes_api(img_path, root_url, face_detect=face_detect)
            check_api_health(root_url=root_url)
        except ConnectionError:
            print("Connection Error - Pausing all submissions")
            pause_required[0] = True
            raise  # Reraising to catch in the main loop

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for _ in range(num_images):
            if pause_required[0]:
                print("Pausing due to a connection error")
                time.sleep(1)
                pause_required[0] = False  # Reset the flag
            future = executor.submit(execute_api_call, img_path, root_url)
            futures.append(future)

            # Optionally, introduce a small delay between submissions if needed
            # time.sleep(0.01)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()

            except ConnectionError:
                # Handle known exceptions here, if any additional action is needed
                pass
            except Exception as exc:
                print(f"An exception occurred: {exc}")


@timeit
def concurrent_healthcheck(root_url, num_images=100):
    """
    Perform concurrent face recognition API calls.
    """
    pause_required = [False]  # Using a list for a mutable, shared state

    def execute_api_call(root_url):
        nonlocal pause_required
        try:
            check_api_health(root_url=root_url)
        except ConnectionError:
            print("Connection Error - Pausing all submissions")
            pause_required[0] = True
            raise  # Reraising to catch in the main loop

    with concurrent.futures.ThreadPoolExecutor(max_workers=2000) as executor:
        futures = []
        for _ in range(num_images):
            if pause_required[0]:
                print("Pausing due to a connection error")
                time.sleep(1)
                pause_required[0] = False  # Reset the flag
            future = executor.submit(execute_api_call, root_url)
            futures.append(future)

            # Optionally, introduce a small delay between submissions if needed
            # time.sleep(0.01)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()

            except ConnectionError:
                # Handle known exceptions here, if any additional action is needed
                pass
            except Exception as exc:
                print(f"An exception occurred: {exc}")

# Define the API endpoint


def check_api_health(root_url):
    try:
        url = f"{root_url}/v1/faceservice/liveness/detail"
        # Make a POST request to the API endpoint
        response = requests.post(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            health_data = response.json()
            # Extract health check information
            cpu_usage = health_data.get('cpu_usage')
            memory_usage = health_data.get('memory_usage')
            disk_usage = health_data.get('disk_usage')
            localhost_resolution = health_data.get('localhost_resolution')
            localhost_port = health_data.get('port')
            
            # Print or process the health check information
            print("CPU Usage:", cpu_usage)
            print("Memory Usage (MB):", memory_usage)
            print("Disk Usage (%):", disk_usage)
            print("Localhost Resolution:", localhost_resolution)
            print("localhost_port:", localhost_port)
            logger.debug(msg=f"health_data: {health_data}")
            
            # You can return this information if needed
            return health_data
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


if __name__ == "__main__":
    connection_config_path = "config/connection.yml"
    connection_config = read_config(path=connection_config_path)     

    img_path = 'image_test/realone.jpg'
    # root_url = "http://192.168.1.28:8091" #Bao
    root_url = "http://127.0.0.1:8090" #Bao
    test_liveness_bytes_api(img_path=img_path, root_url=root_url, face_detect=True)
    test_liveness_base64_api(img_path=img_path, root_url=root_url, face_detect=True)
    concurrent_face_recognition(img_path=img_path, root_url=root_url, face_detect=True, num_images=50)
    check_api_health(root_url=root_url)
    # concurrent_healthcheck(root_url='http://127.0.0.1:8090', num_images=50)


