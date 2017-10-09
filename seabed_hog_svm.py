#/usr/bin/python3.4
import cv2
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from common import SVM, get_hog

COMMON_W = 20
COMMON_H = 20
CLASSES_N = 1

# Paths to images
path_to_images_wheels = './learn_data/seabed/starboard/*.png'
path_to_images_backs = './learn_data/seabed/not_seabed/*.png'


def read_all_images(dirname):
    files=glob.glob(dirname)
    vec_of_images = []
    for file in files:
        img = cv2.imread(file)
        vec_of_images.append(img)
    return vec_of_images


def normalize_images(images):
    images_norm = []
    for img in images:
        if img is not None:
            img_norm = cv2.resize(img, (COMMON_W, COMMON_H), interpolation=cv2.INTER_CUBIC)
            images_norm.append(img_norm)
    return images_norm


def evaluate_model( model, objects, samples, labels ):
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
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return vis


# Read dataset
images_obj = read_all_images(path_to_images_wheels)
images_backs = read_all_images(path_to_images_backs)

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

# Set HOG parameters
hog = get_hog()

# Compute HOG descriptors for each image
hog_descriptors = []
for img in images:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hog_descriptors.append(hog.compute(img_gray))
hog_descriptors = np.squeeze(hog_descriptors)

# Split the dataset to train and test_rawdata
train_n = int(0.9 * len(hog_descriptors))
data_train, data_test = np.split(images, [train_n])
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
labels_train, labels_test = np.split(labels, [train_n])

# Train SVM classifier
print('Training SVM model ...')
model = SVM()
model.train(hog_descriptors_train, labels_train)

print('Saving SVM model ...')
model.save('seabed_svm.dat')

# Test SVM classifier
vis = evaluate_model(model, data_test, hog_descriptors_test, labels_test)
