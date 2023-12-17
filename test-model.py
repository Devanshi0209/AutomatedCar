import os
import time
import tensorflow as tf 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_DIR = r'D:/WindowsNoEditor/PythonAPI/examples/traffic_lights'
IMAGE_PATHS = []

for file in os.listdir(IMAGE_DIR):
    #print(file)
    if file.endswith(".jpeg"):
        IMAGE_PATHS.append(os.path.join(IMAGE_DIR, file))

model_traffic=tf.saved_model.load("D:/WindowsNoEditor/PythonAPI/examples/workspace/exported-models/my_model_ssd_new/saved_model")
PATH_TO_LABELS = "D:/WindowsNoEditor/PythonAPI/examples/workspace/data/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)



for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')
    image_np = np.array(cv2.imread(image_path))
    #cv2.imshow("res",image_np)
    #cv2.waitKey(0)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model_traffic(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    #print(num_detections)
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=1,
          min_score_thresh=.0010,
          agnostic_mode=False)

    #print([category_index.get(value) for index,value in enumerate(detections['detection_classes']) if detections['detection_scores'][index] > 0.1])
    cv2.imshow("res",image_np_with_detections)
    cv2.waitKey(0)


