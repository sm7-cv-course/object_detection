import cv2
import numpy as np
import glob
import argparse
import sklearn
from sklearn.preprocessing import StandardScaler as skStandardScaler
import pickle
import matplotlib.pyplot as plt
from common import SVM, get_hog
from mathlib.standardization import pyStandardScaler

COMMON_W = 20
COMMON_H = 20
MAX_SET_SIZE = 10000

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positive", required=True, help="Path to '*.png' images with objects.")
ap.add_argument("-n", "--negative", required=True, help="Path to '*.png' the images with negative class.")
ap.add_argument("-m", "--model", required=True, help="Trained classifier params.")
args = vars(ap.parse_args())

# Paths to images for learning.
path_to_images_positive = args["positive"] + '/*.png'
path_to_images_negative = args["negative"] + '/*.png'


def read_all_images(dirname):
    files = glob.glob(dirname)
    vec_of_images = []
    count = 0
    for file in files:
        img = cv2.imread(file)
        vec_of_images.append(img)
        count = count + 1
        # Do not load everything.
        if count > MAX_SET_SIZE:
            break
    return vec_of_images


def normalize_images(images):
    images_norm = []
    for img in images:
        if img is not None:
            img_norm = cv2.resize(img, (COMMON_W, COMMON_H), interpolation=cv2.INTER_CUBIC)
            images_norm.append(img_norm)
    return images_norm


def evaluate_model(model, objects, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(objects, resp == labels):
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return vis


# Read dataset
images_obj = read_all_images(path_to_images_positive)
images_backs = read_all_images(path_to_images_negative)

# Normalize images (both train and test sets):
# 1) bring to common size
# 2) histogram equalization - ?
images_obj_norm = normalize_images(images_obj)
images_backs_norm = normalize_images(images_backs)

images_list_norm = images_obj_norm + images_backs_norm
labels = np.int32(np.hstack((np.ones(len(images_obj_norm)), np.zeros(len(images_backs_norm)))))

print('Shuffle data ... ')
# Shuffle data
rand = np.random.RandomState(10)
shuffle = rand.permutation(len(images_list_norm))
images, labels = np.asarray(images_list_norm)[shuffle], labels[shuffle]

# Get HOG parameters.
hog = get_hog()

# Compute HOG descriptors for each image.
hog_descriptors = []
for img in images:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hog_descriptors.append(hog.compute(img_gray))
hog_descriptors = np.squeeze(hog_descriptors)

# Compute HOG for positive set only.
hog_descriptors_positive = []
for (img, lbl) in zip(images, labels):
    if lbl == 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        hog_descriptors_positive.append(hog.compute(img_gray))
hog_descriptors_positive = np.squeeze(hog_descriptors_positive)

# Split the dataset to train and test_rawdata.
train_n = int(0.9 * len(hog_descriptors))
data_train, data_test = np.split(images, [train_n])
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
labels_train, labels_test = np.split(labels, [train_n])

print("Standardize training and testing sets by hand...")
sc = pyStandardScaler()
sc.fit_std(hog_descriptors_train)
# sc.fit_std(hog_descriptors_positive)
hog_descr_train_std_my = sc.standardize(hog_descriptors_train)
hog_descr_test_std_my = sc.standardize(hog_descriptors_test)
sc.save('my_sc.dat')

# Train SVM classifier.
print('Training SVM model ...')
model = SVM()
# model.train(hog_descriptors_train, labels_train)
model.train(hog_descr_train_std_my, labels_train)

print('Saving SVM model ...')
model.save(args["model"])

# Test SVM classifier.
# vis = evaluate_model(model, data_test, hog_descriptors_test, labels_test)
vis = evaluate_model(model, data_test, hog_descr_test_std_my, labels_test)
