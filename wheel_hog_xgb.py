import cv2
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import xgboost as xgb

from common import get_hog

COMMON_W = 20
COMMON_H = 20
CLASSES_N = 1

# Paths to images
path_to_images_wheels = './learn_data/wheels/objs/*.png'
path_to_images_backs = './learn_data/wheels/backs/*.png'

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
        w,h = img.shape[:2]
        fx = COMMON_W / w
        fy = COMMON_H / h
        img_norm = cv2.resize(img, (COMMON_W, COMMON_H), interpolation=cv2.INTER_CUBIC)
        images_norm.append(img_norm)
    return images_norm

def evaluate_model(model, samples, objects, label):
    # make prediction
    resp = model.predict(samples)
    err = (label != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(label, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(objects, resp == label):
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0

        vis.append(img)
    return vis

# Read dataset
images_list_obj = read_all_images(path_to_images_wheels)
images_list_backs = read_all_images(path_to_images_backs)

images_list = images_list_obj + images_list_backs
labels = np.int32(np.hstack((np.ones(len(images_list_obj)), np.zeros(len(images_list_backs)))))

print('Shuffle data ... ')
# Shuffle data
rand = np.random.RandomState(10)
shuffle = rand.permutation(len(images_list))
images, labels = np.asarray(images_list)[shuffle], labels[shuffle]

# Normalize iamges (both train and test sets):
# 1) bring to common size
# 2) histogram equalization - ?
images_norm = normalize_images(images)

# Set HOG parameters
hog = get_hog()

# Compute HOG descriptors for each image
hog_descriptors = []
for img in images_norm:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hog_descriptors.append(hog.compute(img_gray))
hog_descriptors = np.squeeze(hog_descriptors)

# Split the dataset to train and test_rawdata
train_n = int(0.9*len(hog_descriptors))
data_train, data_test = np.split(images_norm, [train_n])
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
labels_train, labels_test = np.split(labels, [train_n])

dtrain = xgb.DMatrix(hog_descriptors_train, label=labels_train)
dtest = xgb.DMatrix(hog_descriptors_test, label=labels_test)
#dtrain.save_binary('testdata.txt')
# specify parameters via map
param = {'max_depth':5, 'eta':0.5, 'objective':'binary:logistic', 'gamma':1.0}
num_round = 5
# Train XGB classifier
print('Training XGB model ...')
model_xgb = xgb.train(param, dtrain, num_round)

print('Saving XGB model ...')
model_xgb.save_model('seal_xgb.dat')

# Test XGB classifier
vis = evaluate_model(model_xgb, dtest, data_test, labels_test)