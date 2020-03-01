import argparse
import predict_utils
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Image Classifier Prediction Program
# JC 2.29.2020
# ===================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Flower Image Predictor'
    )
    # Add the arguments to the argument parser
    # At a minimum, the image path and model path are required
    parser.add_argument('img_path', action='store',
                        help='Path to the test image')
    parser.add_argument('model_path', action='store',
                        help='Path to the Keras model')
    parser.add_argument('--top_k', action='store', dest='k', type=int,
                        help='Number of prediction classes to return')
    parser.add_argument('--category_names', action='store',
                        dest='classnames_path',
                        help='Path to a JSON file containing classnames')
    arguments = parser.parse_args()

    # Load the tensorflow model
    # Reference: https://github.com/tensorflow/tensorflow/issues/26835
    model = tf.keras.models.load_model(
        arguments.model_path,
        custom_objects={'KerasLayer':hub.KerasLayer}
        )

    # Load the image
    image = predict_utils.process_image(arguments.img_path)

    # Make predictions
    results = model.predict(image)
    results = results[0]

    # See if a classname JSON has been passed
    if arguments.classnames_path:
        classnames = predict_utils.read_json(arguments.classnames_path)
    else:
        classnames = None

    # Check for arguments
    # If K has been passed, print the top K classes
    if arguments.k and classnames:
        top_probs, top_classes = predict_utils.k_results(results, arguments.k)
        top_classes = predict_utils.class_name_finder(top_classes, classnames)
        print("Top Probabilities:")
        for i in range(arguments.k):
            print(i + 1, ": Label: ", top_classes[i],
                  ", Probability: ", top_probs[i])
    # If K is passed but not classnames,
    # return the class number and probabilities
    elif arguments.k and classnames==None:
        top_probs, top_classes = predict_utils.k_results(results, arguments.k)
        print("Top Probabilities:")
        for i in range(arguments.k):
            print(i + 1, ": Label: ", top_classes[i],
                  ", Probability: ", top_probs[i])
    # If no K but there are classnames, return the top probability and the name
    elif classnames:
        top_class = np.argmax(results)
        top_classname = classnames[str(top_class + 1)]
        top_prob = results[top_class]
        print("Top Probability:")
        print("Label: ", top_classname, ", Probability: ", top_prob)
    # If no K and no classnames, return the top class and probability
    else:
        top_class = np.argmax(results)
        top_prob = results[top_class]
        print("Top Probability:")
        print("Label: ", top_class, ", Probability: ", top_prob)
