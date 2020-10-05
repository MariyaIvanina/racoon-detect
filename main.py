from flask_cors import CORS
from waitress import serve
from flask import Flask, request, Response
import numpy as np
from object_detection.utils import label_map_util, visualization_utils
import tensorflow as tf
import json
import cv2
from PIL import Image

class GenericEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)

def detect_for_image(image_np, with_image=False):

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    threshold = 0.5
    detected_results = []
    for box, _class, score in zip(
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores']):
        if score >= threshold:
            detected_results.append({
              "class_name": category_index[_class]["name"],
              "probability": score,
              "box": list(np.asarray(box, dtype=np.float))
            })
    image_np_with_detections = image_np.copy()

    visualization_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False,
          line_thickness=10)
    if with_image:
        return {
          "image_with_detections": image_np_with_detections,
          "detected_results": detected_results
        }
    return {
          "detected_results": detected_results
        }

def create_application():
    application = Flask(__name__)
    return application

print("Started initialization")
path_to_model = "trained_inference_graph/saved_model"
label_map_path = "label_map.pbtxt"
detect_fn = tf.saved_model.load(path_to_model)
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
print("Initialized the model")

application = create_application()
CORS(application)

def prepare_response(detected_results):
    response_encoded = json.dumps(detected_results, cls=GenericEncoder)
    return Response(response=response_encoded, status=200, mimetype="application/json")

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def resize_image(image, basewidth=640):
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)
    return image_np

@application.route('/object_detection_photo', methods=['POST'])
def detect_racoons_photo():
    nparr = np.frombuffer(request.data, np.uint8)
    imageBGR = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_np)
    img = resize_image(img, basewidth=640)
    print("Got an image with the shape ", img.shape)
    return prepare_response(detect_for_image(img, with_image=True))

@application.route('/object_detection_video_frame', methods=['POST'])
def detect_racoons_video_frame():
    img = np.asarray(json.loads(request.data), dtype=np.uint8)
    print("Got an image with the shape ", img.shape)
    return prepare_response(detect_for_image(img))

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)