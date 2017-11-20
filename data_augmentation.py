"""
Data augmentation by mirroring. 
"""
# Import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import matplotlib.pyplot as plt
import glob

from common import SVM, get_hog
sys.path.insert(0, './pyimagesearch')
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the images to augment.")
ap.add_argument("-m", "--mirror", required=False, default='hv',
                help="Direction of mirroring: 'v' - vertical, 'h' - horizontal, 'hv' or 'vh' - both.")
args = vars(ap.parse_args())


def read_all_images(dirname):
    """
    Read images and create map {falename, image}.
    :param dirname: directory storing images;
    :return: dictionary {falename, image}.
    """
    files = glob.glob(dirname)
    vec_of_images = []
    vec_of_filenames = []
    for file in files:
        img = cv2.imread(file)
        vec_of_images.append(img)
        vec_of_filenames.append(file)
    return zip(vec_of_filenames, vec_of_images)


# Read all images
images_dict = read_all_images(args['images'] + '/*.png')

vert = False
hor = False
flip = -1
if args['mirror'] == 'v':
    vert = True
    flip = 1
elif args['mirror'] == 'h':
    hor = True
    flip = 0
else:
    hor = True
    ver = True

for (key, value) in images_dict:
    # Flip image.
    mir_img = cv2.flip(value, flip)
    # Change file name.
    fname = key.partition('.')[0] + '_flip.' + key.partition('.')[2]
    # Save mirrored image.
    cv2.imwrite(fname, mir_img)
