# TensorFlow Image Classifier

## About
This project is an image classifier built using the Oxford Flowers 102 dataset.  This dataset contains examples of 102 different kinds of flowers.  This project aims to identify these 102 types of flowers by utilizing the MobileNet neural network.

## Prerequisites

This project runs in Python 3.  Python may be downloaded from:

https://www.python.org/downloads/

To install, run the downloaded executable.

This project makes use of libraries managed by Conda.  Information on downloading and installing Conda may be found at:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Part of this project is formatted inside of a Jupyter Notebook.  Information on downloading and installing may be found at:

https://jupyter.org/install

This project uses SciPy packages to perform calculations.  These packages include NumPy and Matplotlib.  Information on installing these packages may be found at:

https://scipy.org/install.html

The classifier requires TensorFlow and TensorFlow Hub to properly run.  Information on downloading and installing these packages may be found at:

https://www.tensorflow.org/hub/installation

## Running

### Jupyter Notebook
To view the project notebook that was used to develop the Keras model, navigate to the directory with the Project_Image_Classifier_Project.ipynb file.  Run the following command:

		jupyter notebook

This will start the jupyter notebook software.  From there, a window will open in your default web browser.  Double click on the Project_Image_Classifier_Project.ipynb file, and you will be able to view the code and results for this portion of the project.  Alternatively, you may navigate to the project files and double click on Project_Image_Classifier_Project.html to view the project code and results in a read-only form.

### Command Line Tool
Running the command line tool requires the following files:
predict.py
predict_utils.py
model.h5

To run the tool, run the following command

		python3 predict.py path/to/image model.h5 --top_k K --category_names label_map.json

Where:
* path/to/image: A test image to be classified
* model.h5: A Keras model file.  This is intended to be the model.h5 file developed for this project.
* --top_k: (Optional) An integer value which indicates how many top probabilities to print.  For example, --top_k 5 would return the top 5 classes and their probabilities
* --category_names: (Optional) A JSON file which indicates the names of each label in the prediction.  If this file is provided, then the name of each category will be printed along with its probability.
