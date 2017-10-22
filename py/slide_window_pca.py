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
# from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pickle
from common import SVM, get_hog

sys.path.insert(0, './../pyimagesearch')
from helpers import pyramid
from helpers import sliding_window

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

# Set type of classifier.
model_type = args["classifier"]
if model_type is None:
    model_type = 'svm'

# Start time measurement.
e1 = cv2.getTickCount()

# Load pre trained classifier.
if model_type == 'svm':
    model = SVM()
    model.load(args["model"])
elif model_type == 'rf':
    model = joblib.load(args["model"])

# Initialize the HOG descriptor/person detector.
hog = get_hog()

# Get PCA parameters.
table_path = args["pca"]
if table_path is None:
    table_path = './../dat/pca.src'

print("Load table {eigenvalues, eigenvectors}...")
with open(table_path, 'rb') as f:
    table = pickle.load(f)
eigenvalues, eigenvectors = zip(*table)
# eigenvalues = np.array(eigenvalues)
eigenvectors = np.array(eigenvectors)

# Loop over the image pyramid.
dwnsmpl_scale = 1.5
cur_scale = 1
for resized in pyramid(img_gray, scale=dwnsmpl_scale, do_pyramid=True):
    # Loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winW, winH)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW

        hog_descriptor = hog.compute(window)
        hog_descriptor = np.squeeze(hog_descriptor)
        hog_descriptor = np.reshape(hog_descriptor, (1, hog_descriptor.shape[0]))
        hog_descriptor = (hog_descriptor.dot(eigenvectors[0:len(eigenvalues)].T)).astype(np.float32)
        one_resp = model.predict(hog_descriptor)
        if one_resp == 1:
            lup_x = int(x * cur_scale)
            lup_y = int(y * cur_scale)
            rbot_x = int((x + winW) * cur_scale)
            rbot_y = int((y + winH) * cur_scale)
            cv2.rectangle(img_gray, (lup_x, lup_y), (rbot_x, rbot_y), 255, 2)
    cur_scale *= dwnsmpl_scale

# End time measurement.
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Detection process time = ", time)

plt.imshow(img_gray)
plt.show()
