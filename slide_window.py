# Import the necessary packages
import numpy as np
import argparse
import pickle
import cv2
import sys
import matplotlib.pyplot as plt
from mathlib.standardization import pyStandardScaler

from common import SVM, get_hog
sys.path.insert(0, './pyimagesearch')
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-m", "--model", required=True, help="'.dat' file with trained classifier params.")
ap.add_argument("-o", "--out", required=False, help="Output directory.")
args = vars(ap.parse_args())

# Load the image and trained classifier.
image_orig = cv2.imread(args["image"])
dwnsmpl_rate = 0.25
img_dwnsmpl = cv2.resize(image_orig, None, fx=dwnsmpl_rate, fy=dwnsmpl_rate, interpolation=cv2.INTER_NEAREST)
img_gray = cv2.cvtColor(img_dwnsmpl, cv2.COLOR_RGBA2GRAY)

# Define the window width and height.
(winW, winH) = (20, 20)

# Load pretrained classifier.
model = SVM()
model.load(args["model"])

# Initialize the HOG descriptor/person detector.
hog = get_hog()

print("Load standard scaler...")
path_to_sc = 'sc.dat'
try:
    with open(path_to_sc, 'rb') as f:
        sc = pickle.load(f)
except:
    sc = None

my_sc = pyStandardScaler()
my_sc.load('my_sc.dat')

# Loop over the image pyramid.
print("Sliding window loop...")
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
        # hog_descriptor = np.squeeze(hog_descriptor)
        # hog_descriptor = np.reshape(hog_descriptor, (1, hog_descriptor.shape[0]))
        # hog_descriptor = sc.transform(hog_descriptor.T)
        hog_descriptor = my_sc.standardize(hog_descriptor.T)
        one_resp = model.predict(hog_descriptor)
        # one_resp = model.predict(hog_descriptor_std)
        if one_resp == 1:
            lup_x = int(x * cur_scale)
            lup_y = int(y * cur_scale)
            rbot_x = int((x + winW) * cur_scale)
            rbot_y = int((y + winH) * cur_scale)
            cv2.rectangle(img_gray, (lup_x, lup_y), (rbot_x, rbot_y), 255, 2)
    cur_scale *= dwnsmpl_scale

plt.imshow(img_gray)
plt.show()

if args["out"] is not None:
    f_name = args["image"].split('/')[-1]
    name = f_name.split('.')[0]
    out_path = args["out"] + '/' + name + '_rez_std.png'
    cv2.imwrite(out_path, img_gray)
