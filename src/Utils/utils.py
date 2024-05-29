from functools import wraps
import time
from datetime import datetime
import os
import json
import yaml
import numpy as np
import cv2
import base64

import socket
import shutil
import psutil


    
def check_localhost():
    localhost = socket.gethostbyname('localhost')
    return localhost

def check_disk_usage(disk):
    du = shutil.disk_usage(disk)
    free = du.free / du.total * 100
    return free 

def check_memory_usage():
    mu = psutil.virtual_memory().available
    total = mu / (1024.0 ** 2)
    return total

def check_cpu_usage():
    usage = psutil.cpu_percent(1)
    return usage

def convert_img_to_base64(img):
    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)  # Adjust format as needed (e.g., '.png')
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return base64_img


def convert_img_path_to_base64(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to read image from {image_path}")
        base64_image = convert_img_to_base64(img)
        return base64_image

    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def bytes_to_image(image_bytes):
  """
  Converts a bytes object containing image data to an OpenCV image.

  Args:
      image_bytes (bytes): The bytes object representing the image data.

  Returns:
      numpy.ndarray: The OpenCV image as a NumPy array.

  Raises:
      ValueError: If the image data is invalid.
  """
  # Convert the bytes object to a NumPy array
  nparr = np.frombuffer(image_bytes, np.uint8)

  # Decode the image using cv2.imdecode()
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Check for decoding errors
  if img is None:
    raise ValueError("Invalid image data")

  return img

    
def convert_base64_to_img(base64_image:str):
    image_bytes = base64.b64decode(base64_image)
    img = bytes_to_image(image_bytes)
    return img

def refine(boxes, max_width, max_height, shift=0.1):
    """
    Refine the face boxes to suit the face landmark detection's needs.

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


def convert_ms_to_hms(ms):
    seconds = ms / 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    seconds = round(seconds, 2)
    
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        time_delta = convert_ms_to_hms(total_time*1000)

        print(f'{func.__name__.title()} Took {time_delta}')
        return result
    return timeit_wrapper

def is_file(path:str):
    return '.' in path

def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
        return path
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")
        return path

def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def base64_path_to_image(base64_path = 'image_test/fakeios.txt'):
    with open(base64_path, 'r') as file:
        base64_string = file.read().strip()
    image_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def sort_coordinates(coords):
    startX, startY, endX, endY = coords
    startX, endX = sorted([startX, endX])
    startY, endY = sorted([startY, endY])
    return startX, startY, endX, endY

def draw_image(frame, pred_class, prob, location):
    
    startX, startY, endX, endY = location
    if pred_class[0] == 1:
        text = 'real'
        color = (0,255,0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, f"liveness prob: {round(100*prob[1], 2)}", (startX-10, startY-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    else:
        text = 'fake'
        color = (255,0,0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, f"liveness prob: {round(100*prob[1], 2)}", (startX-10, startY-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
    return frame
