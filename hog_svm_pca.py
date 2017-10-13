import cv2
import numpy as np
import glob
import argparse
import pickle
from common import SVM, get_hog

COMMON_W = 20
COMMON_H = 20

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positive", required=True, help="Path to '*.png' images with objects.")
ap.add_argument("-n", "--negative", required=True, help="Path to '*.png' the images with negative class.")
ap.add_argument("-m", "--model", required=True, help="Trained classifier params.")
args = vars(ap.parse_args())

# Paths to images for learning
path_to_images_positive = args["positive"] + '/*.png'
path_to_images_negative = args["negative"] + '/*.png'


def read_all_images(dirname):
    files = glob.glob(dirname)
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
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return vis


def pca_projection(pcaset, maxComponents, testset, compressed):
    cv2.PCA(pcaset, cv2.Mat(), cv2.PCA_DATA_AS_ROW, maxComponents)


# Read dataset
images_obj = read_all_images(path_to_images_positive)
images_backs = read_all_images(path_to_images_negative)

# Normalize images (both train and test sets):
# 1) bring to common size
# 2) histogram equalization - ?
images_obj_norm = normalize_images(images_obj)
images_backs_norm = normalize_images(images_backs)

# Get HOG parameters
hog = get_hog()

# Compute features for positives.
hog_descriptors_obj = []
for img in images_obj_norm:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hog_descriptors_obj.append(hog.compute(img_gray))
hog_descriptors_obj = np.squeeze(hog_descriptors_obj)

# Compute features for negative sample.
hog_descriptors_neg = []
for img in images_backs_norm:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hog_descriptors_neg.append(hog.compute(img_gray))
hog_descriptors_neg = np.squeeze(hog_descriptors_neg)

# Save features to external analysis.
## Prepare.
labels = np.int32(np.hstack((np.ones(len(images_obj_norm)), np.zeros(len(images_backs_norm)))))
hogs = np.vstack((hog_descriptors_obj, hog_descriptors_neg))
table = list(zip(labels, hogs))
## Write whole list to file.
with open('hog_descriptors.src', 'wb') as f:
    pickle.dump(table, f)

# Perform PCA on positive sample.
mean, eigenvectors = cv2.PCACompute(hog_descriptors_obj, np.mean(hog_descriptors_obj, axis=0).reshape(1, -1))



# Apply PCA to positive sample

# Apply PCA to negative sample



# images_list_norm = images_obj_norm + images_backs_norm
# labels = np.int32(np.hstack((np.ones(len(images_obj_norm)), np.zeros(len(images_backs_norm)))))
#
# print('Shuffle data ... ')
# # Shuffle data
# rand = np.random.RandomState(10)
# shuffle = rand.permutation(len(images_list_norm))
# images, labels = np.asarray(images_list_norm)[shuffle], labels[shuffle]
#
# # Get HOG parameters
# hog = get_hog()
#
# # Compute HOG descriptors for each image
# hog_descriptors = []
# for img in images:
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
#     hog_descriptors.append(hog.compute(img_gray))
# hog_descriptors = np.squeeze(hog_descriptors)
#
# # Split the dataset to train and test_rawdata
# train_n = int(0.9 * len(hog_descriptors))
# data_train, data_test = np.split(images, [train_n])
# hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
# labels_train, labels_test = np.split(labels, [train_n])
#
# # Train SVM classifier
# print('Training SVM model ...')
# model = SVM()
# model.train(hog_descriptors_train, labels_train)
#
# print('Saving SVM model ...')
# model.save(args["model"])
#
# # Test SVM classifier
# vis = evaluate_model(model, data_test, hog_descriptors_test, labels_test)
