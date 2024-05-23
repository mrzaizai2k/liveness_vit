import json
import sys
sys.path.append("")
import socket
import time
from contextlib import contextmanager
from multiprocessing import Process
import torch 
from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata, TopicPartition
from src.Utils.utils import *
from src.kafka.utils import *
from src.model import VisionTransformerModel, ResnetModel


face_detector = FaceDetection(model_config_path="config/face_detection.yml")
liveness_model = VisionTransformerModel(model_config_path="config/vit_inference.yml")


class PredictFrames(Process):

    def __init__(self,
                 topic,
                 verbose=False,
                 group=None,
                 target=None,
                 name=None,
                 kafka_host = '0.0.0.0',
                 kafka_port = '9092',
                 show_image = False,):
        """
        FACE MATCHING TO QUERY FACES --> Consuming frame objects to produce predictions.

        :param processed_frame_topic: kafka topic to consume from stamped encoded frames with face detection and encodings.
        :param query_faces_topic: kafka topic which broadcasts query face names and encodings.
        :param scale: (0, 1] scale used during pre processing step.
        :param verbose: print log
        :param rr_distribute: use round robin partitioner and assignor, should be set same as respective producers or consumers.
        :param group: group should always be None; it exists solely for compatibility with threading.
        :param target: Process Target
        :param name: Process name
        """
        super().__init__(group=group, target=target, name=name)

        self.iam = "{}-{}".format(socket.gethostname(), self.name)
        self.topic = topic
        self.verbose = verbose
        self.kafka_host = kafka_host
        self.kafka_port = kafka_port
        self.show_image= show_image
        print("[INFO] I am ", self.iam)

    def run(self):
        """Consume pre processed frames, match query face with faces detected in pre processing step
        (published to processed frame topic) publish results, box added to frame data if in params,
        ORIGINAL_PREFIX == PREDICTED_PREFIX"""

        frame_consumer = KafkaConsumer(group_id='face_recognition',
                                       bootstrap_servers=[f"{self.kafka_host}:{self.kafka_port}"],
                                       key_deserializer=lambda key: key.decode(),
                                       value_deserializer=lambda value: json.loads(value.decode()),
                                       auto_offset_reset="earliest",
                                       )

        frame_consumer.subscribe([self.topic])
        try:
            while True:
                if self.verbose:
                    print("[PredictFrames {}] WAITING FOR NEXT FRAMES..".format(socket.gethostname()))

                raw_frame_messages = frame_consumer.poll(timeout_ms=10, max_records=10)

                for topic_partition, msgs in raw_frame_messages.items():
                    # Get the predicted Object, JSON with frame and meta info about the frame
                    print('msgs', len(msgs))
                    for msg in msgs:
                        # print('msg', msg)
                        tp = TopicPartition(msg.topic, msg.partition)
                        offsets = {tp: OffsetAndMetadata(msg.offset, None)}
                        frame_consumer.commit(offsets=offsets)

                        # print(f'partition: {msg.partition}, offset: {msg.offset}, timestamp: {msg.timestamp}')
                        # base64_img = msg.value["original_frame"]
                        # print('base64_img', base64_img)

                        np_array = np_from_json(msg.value, prefix_name="original") 
                        
                        face_locations, frame = face_detector.predict(np_array)
                        
                        for startX, startY, endX, endY in face_locations:
                   
                            refined =  refine([[startX, startY, endX, endY]], max_height=frame.shape[0], max_width=frame.shape[1])[0]
                            startX, startY, endX, endY = refined[:4].astype(int)
                            face = frame[startY:endY, startX:endX]
                            pred_class , prob = liveness_model.predict(face)

                            camera = msg.value["camera"]

                            print(f"Result in camera {camera} is {pred_class} with liveness prob: {prob[1]}")
                            if self.show_image:
                                frame = frame.copy()
                                draw_image(frame, pred_class=pred_class, prob=prob, location=[startX, startY, endX, endY])


                        # save_result(camera=camera, response=result)
                        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        if self.show_image:

                            cv2.imshow('img', frame)      
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break          
        

        except KeyboardInterrupt as e:
            print("Closing Stream")
            frame_consumer.close()
            if str(self.name) == "1":
                pass

        finally:
            print("Closing Stream")
            frame_consumer.close()


@contextmanager
def timer(name):
    """Util function: Logs the time."""
    t0 = time.time()
    yield
    print("[{}] done in {:.3f} s".format(name, time.time() - t0))

if __name__=="__main__":
    kafka_config = read_config("config/kafka.yml")
    kafka_host = kafka_config['kafka_host']
    kafka_port = kafka_config['kafka_port']
    kafka_topic = kafka_config['kafka_topic']

    consumer = PredictFrames(name="video-1",topic=kafka_topic, kafka_host = kafka_host, kafka_port=str(kafka_port), show_image = True)
    consumer.run()