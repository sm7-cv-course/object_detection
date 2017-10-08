# Import the necessary packages
import numpy as np
import argparse
import time
import cv2
import sys
import matplotlib.pyplot as plt

from common import SVM, get_hog
sys.path.insert(0, './pyimagesearch')
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--model", required=True, help="Trained classifier params")
args = vars(ap.parse_args())

# Load the image and trained classifier
image_orig = cv2.imread(args["image"])
img_gray = cv2.cvtColor(image_orig, cv2.COLOR_RGBA2GRAY)

# Define the window width and height
(winW, winH) = (48, 48)

# Load pretrained classifier
model = SVM()
model.load(args["model"])

# Initialize the HOG descriptor/person detector
hog = get_hog();

# Loop over the image pyramid
dwnsmpl_scale = 1.5
cur_scale = 1
for resized in pyramid(img_gray, scale=dwnsmpl_scale, do_pyramid=True):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        # print(x,y,window.shape[0],window.shape[1])
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW

        hog_descriptor = hog.compute(window)
        hog_descriptor = np.squeeze(hog_descriptor)
        hog_descriptor = np.reshape(hog_descriptor, (1, hog_descriptor.shape[0]))
        # hog_descriptor = np.concatenate(hog_descriptor,hog_descriptor)
        one_resp = model.predict(hog_descriptor)
        if one_resp == 1:
            lup_x = int(x * cur_scale)
            lup_y = int(y * cur_scale)
            rbot_x = int((x + winW) * cur_scale)
            rbot_y = int((y + winH) * cur_scale)
            cv2.rectangle(img_gray, (lup_x, lup_y), (rbot_x, rbot_y), 255, 2)
    cur_scale *= dwnsmpl_scale

plt.imshow(img_gray)
plt.show()
