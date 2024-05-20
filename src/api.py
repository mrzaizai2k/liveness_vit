import sys
sys.path.append("")

from flask import Flask, request, jsonify
import timm
import torch
from torch import nn
from torchvision import transforms
import io
from src.Utils.utils import *
from src.Utils.inference_utils import *

# Initialize Flask app
app = Flask(__name__)

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_config_path= "config/vit_inference.yml"
connection_config_path = "config/connection_config.yaml"

connection_config = read_config(path=connection_config_path)
model_config = read_config(path=model_config_path)


MODEL_DIR = model_config.get("MODEL_DIR")

# Load the model
print('LOADING THE LIVENESS MODEL...')
model, transform = initialize_model(model_path=MODEL_DIR, device=device)



# Define the route for liveness detection
@app.route('/v1/faceservice/liveness/predict', methods=['POST'])
def predict_liveness():
    try:
        
        if request.content_type == 'application/json':  # Base64 encoded image
            image_data = request.json['img']
            image_data = base64.b64decode(image_data)

        elif request.content_type == 'image/jpeg':
            image_data = request.data
    
        
        if image_data:
            img_bytes = image_data
            prediction = predict_image(img_bytes, model=model, transform=transform, device=device)
            return jsonify({"liveness": "real" if prediction == 1 else "fake"}), 200
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def main():
    
    LIVENESS_HOST = connection_config.get("LIVENESS_HOST")
    LIVENESS_PORT = connection_config.get("LIVENESS_PORT")

    app.run(host=LIVENESS_HOST, port=str(LIVENESS_PORT))


if __name__ == "__main__":
    main()
