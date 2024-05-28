import sys
sys.path.append("")
import numpy as np
import base64
import cv2 

from flask import Flask, request, jsonify
from waitress import serve

from src.Utils.utils import *
from src.Utils.inference_utils import *
from src.model import VisionTransformerModel, ResnetModel

from src.Utils.logger import create_logger
logger = create_logger(logfile="logs/liveness_api.log")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Initialize Flask app
app = Flask(__name__)


model_config_path= "config/vit_inference.yml"
connection_config_path = "config/connection.yml"

connection_config = read_config(path=connection_config_path)
model = VisionTransformerModel(model_config_path=model_config_path)
face_detector = FaceDetection(model_config_path="config/face_detection.yml")
logger.info('Loading models')


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"



@app.route('/v1/faceservice/liveness/detail', methods=['POST'])
def health():
    checks = {
        "cpu_usage": check_cpu_usage(),
        "memory_usage": check_memory_usage(),
        "disk_usage": check_disk_usage('/'),
        "localhost_resolution": check_localhost(),
        "port": request.environ.get('SERVER_PORT')
    }
    logger.debug(msg=f"checks: {checks}")
    
    return jsonify(checks), 200

# Define the route for liveness detection
@app.route('/v1/faceservice/liveness/predict', methods=['POST'])
def predict_liveness():
    try:
        face_detect = request.args.get('face_detect', 'false').lower() == 'true'

        if request.content_type == 'application/json':  # Base64 encoded image
            image_data = request.json['img']
            image_data = base64.b64decode(image_data)

        elif request.content_type == 'image/jpeg':
            image_data = request.data
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
        if face_detect:
            face_locations, img = face_detector.predict(img)
            if face_locations:
                startX, startY, endX, endY = face_locations[0]  # Assuming a single face detection for simplicity
                img = img[startY:endY, startX:endX]
            else:
                data = {"status_code": 401,
                        "message": 'There is no face',
                        }

        # print("image_data", image_data)
        pred_class , prob = model.predict(img)
        score = round(prob[pred_class[0]].item(),3)
        if pred_class[0] == 1:
            data = {"status_code": 200,
                    "liveness": 'real',
                    "score": score}
            logger.debug(msg=f"data: {data}")
        else:
            data = {"status_code": 200,
                    "liveness": 'fake',
                    "score": score}
            logger.debug(msg=f"data: {data}")


        return jsonify(data), 200
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def main():
    
    LIVENESS_HOST = connection_config.get("LIVENESS_HOST")
    LIVENESS_PORT = connection_config.get("LIVENESS_PORT")
    logger.info(msg=f"Host: {LIVENESS_HOST} - Port: {LIVENESS_PORT}")
    # app.run(host=LIVENESS_HOST, port=str(LIVENESS_PORT), debug=True)
    serve(app, host=LIVENESS_HOST, port=LIVENESS_PORT, threads=4, url_scheme='https' , expose_tracebacks=True)


if __name__ == "__main__":
    main()
