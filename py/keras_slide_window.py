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
from keras.models import model_from_yaml
#from sklearn.externals import joblib
import pickle
sys.path.insert(0, '../pyimagesearch')
# import helpers.pyramid
# import helpers.sliding_window
from helpers import pyramid
from helpers import sliding_window


# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-m", "--model", required=True, help="'.dat' file with trained classifier params.")
ap.add_argument("-g", "--generator", required=True, help="File with pre trained generator.")

args = vars(ap.parse_args())

# Load testing image.
image_orig = cv2.imread(args["image"])
img_gray = image_orig
# img_gray = cv2.cvtColor(image_orig, cv2.COLOR_RGBA2GRAY)

# Define initial window width and height.
(winW, winH) = (20, 20)

model = Sequential()

# Load pretrained model.
# model = keras.models.load_model('mobilenet.h5', custom_objects={
#                          'relu6': mobilenet.relu6,
#                          'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

model = keras.models.load_model(args["model"])

# Load pretrained generator
with open(args["generator"], 'rb') as f:
    datagen = pickle.load(f)
    f.close()

dwnsmpl_scale = 1.5
cur_scale = 1

print("Pyramid slide window search...")
for resized in pyramid(img_gray, scale=dwnsmpl_scale, do_pyramid=True):
    # Loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winW, winH)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW

        # Converte to monochromical triplet
        win_gray = cv2.cvtColor(window, cv2.COLOR_RGBA2GRAY)
        win_grey_triplet = cv2.merge((win_gray, win_gray, win_gray))
        window = win_grey_triplet

        # Prepare "batch" for testing.
        windows = []
        windows.append(window)
        windows = np.vstack((windows, windows))

        one_resp = model.predict_generator(generator=datagen.flow(windows), steps=1, max_queue_size=10, workers=10,
                                           use_multiprocessing=False, verbose=0)
        print("one_resp = ", one_resp)
        if one_resp[0, 1] > 0.5:
            lup_x = int(x * cur_scale)
            lup_y = int(y * cur_scale)
            rbot_x = int((x + winW) * cur_scale)
            rbot_y = int((y + winH) * cur_scale)
            cv2.rectangle(img_gray, (lup_x, lup_y), (rbot_x, rbot_y), 255, 2)
    cur_scale *= dwnsmpl_scale

plt.imshow(img_gray)
plt.show()

# prediction = model.predict_generator(generator, steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
