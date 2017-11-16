"""
Train binary classifier from Keras framework.
"""

from __future__ import print_function
import keras
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import argparse
import cv2
import glob
import pickle


# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positive", required=True, help="Path to '*.png' images with objects.")
ap.add_argument("-n", "--negative", required=True, help="Path to '*.png' the images with negative class.")
ap.add_argument("-m", "--model", required=True, help="Trained classifier params.")
args = vars(ap.parse_args())


# Training parameters.
(COMMON_W, COMMON_H) = (20, 20)
batch_size = 32
num_classes = 2
epochs = 10
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
print("save_dir = ", save_dir)
# Paths to images for learning.
path_to_images_positive = args["positive"] + '/*.png'
path_to_images_negative = args["negative"] + '/*.png'
model_name = args["model"]
print("model_name = ", model_name)


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
            img_gray = cv2.cvtColor(img_norm, cv2.COLOR_RGBA2GRAY)
            # img_grey_triplet = (img_gray, img_gray, img_gray)
            img_grey_triplet = cv2.merge((img_gray, img_gray, img_gray))
            images_norm.append(img_grey_triplet)
    return images_norm


# Read dataset.
images_obj = read_all_images(path_to_images_positive)
images_backs = read_all_images(path_to_images_negative)
# Normalize images (both train and test sets):
# 1) bring to common size;
# 2) histogram equalization - ?
images_obj_norm = normalize_images(images_obj)
images_backs_norm = normalize_images(images_backs)

images = np.vstack((images_obj_norm, images_backs_norm))
labels = np.int32(np.hstack((np.ones(len(images_obj_norm)), np.zeros(len(images_backs_norm)))))

# Split the dataset to train and test
train_n = int(0.9 * len(labels))
x_train, x_test = np.split(images, [train_n])
y_train, y_test = np.split(labels, [train_n])

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-8)

# Let's train the model using RMSprop
print("Compile model.")
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # Set input mean to 0 over the dataset.
        samplewise_center=True,  # Set each sample mean to 0.
        featurewise_std_normalization=True,  # Divide inputs by std of the dataset.
        samplewise_std_normalization=True,  # Divide each input by its std.
        zca_whitening=False,  # Apply ZCA whitening (Mahalanobis).
        rotation_range=0,  # Randomly rotate images in the range (degrees, 0 to 180).
        width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width).
        height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height).
        horizontal_flip=False,  # Randomly flip images.
        vertical_flip=False)  # Randomly flip images.

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Write ImageDataGenerator parameters.
    ImgDataGenerator_path = 'datagen_grey.dat'
    with open(ImgDataGenerator_path, 'wb') as f:
        pickle.dump(datagen, f)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
