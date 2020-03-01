import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Prediction Utilities
# JC 2.29.2020
# ====================

def class_name_finder(classes_pred, class_list):
    class_names = []
    for i in range(len(classes_pred)):
        # Find the index of each top class, and get its name
        index = classes_pred[i]
        class_names.insert(i, class_list[str(index + 1)])
    return class_names

def process_image(image_path):
    # Open the image and convert it to a NumPy array
    image_in = Image.open(image_path)
    image_in = np.asarray(image_in)
    # Create a tensor and normalize
    image_out = tf.cast(image_in, tf.float32)
    image_out = tf.image.resize(image_out, ((224, 224)))
    image_out /= 255
    # Return as a NumPy array
    # Source: https://stackoverflow.com/questions/48794214/expected-input-to-have-4-dimensions-but-got-array-with-shape
    return image_out.numpy().reshape(-1, 224, 224, 3)

def k_results(results, k):
    # Strip the results down to the top K results, and their classes
    top_probs = []
    top_classes = []
    for i in range(k):
        index = np.argmax(results)
        top_classes.insert(i, index)
        top_probs.insert(i, results[index])
        results = np.delete(results, [index])
    return top_probs, top_classes

def read_json(json_path):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
        return json_dict
