"""
Pyramid sliding window exhaustive search for object detection.
Works with different binary classifiers.
"""
# Import the necessary packages.
import argparse
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.externals import joblib
import pickle
import helpers.pyramid
import helpers.sliding_window



# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-m", "--model", required=True, help="'.dat' file with trained classifier params.")
ap.add_argument("-c", "--classifier", required=False, help="Type of trained classifier: 'svm' - SVM, "
                                                           "'rf' - random forest. 'svm' by default.")
ap.add_argument("-o", "--pca", required=False,
    help="File with table containing calculated PCA parameters, used in training process.")
args = vars(ap.parse_args())

# Load testing image.
image_orig = cv2.imread(args["image"])
img_gray = cv2.cvtColor(image_orig, cv2.COLOR_RGBA2GRAY)

# Define initial window width and height.
(winW, winH) = (20, 20)

model = Sequential()