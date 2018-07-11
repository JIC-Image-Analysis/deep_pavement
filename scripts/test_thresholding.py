import os
import random

import numpy as np

import keras
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from skimage.filters import threshold_otsu

from imageio import imread, imsave


def load_data_from_disk_images(base_path):
    """Load images found at base_path and convert into x_train and y_train
    arrays suitable for training keras models."""

    wall_root = os.path.join(base_path, "wall")
    wall_filenames = os.listdir(wall_root)
    wall_fpaths_labels = [
        (os.path.join(wall_root, fn), 1)
        for fn in wall_filenames
    ]
    not_wall_root = os.path.join(base_path, "not_wall")
    wall_filenames = os.listdir(not_wall_root)
    not_wall_fpaths_labels = [
        (os.path.join(not_wall_root, fn), 0)
        for fn in wall_filenames
    ]

    fpaths_labels = wall_fpaths_labels + not_wall_fpaths_labels
    random.shuffle(fpaths_labels)

    images = []
    labels = []
    for fpath, label in fpaths_labels:
        images.append(imread(fpath))
        labels.append(label)
    stack = np.transpose(images, (0, 1, 2))
    n_images, xdim, ydim = stack.shape

    # Channels last
    x_train = stack.reshape(n_images, xdim, ydim, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np.array(labels)
    y_train = np_utils.to_categorical(y_train)

    return x_train, y_train


def test_thresholding():

    x_train, y_train = load_data_from_disk_images("data/train")

    cntrs = x_train[:,25,25,0]

    thresh = threshold_otsu(cntrs)

    predictions = cntrs > (0.35 * thresh)

    scores = []
    for n, pred in enumerate(predictions):
        scores.append(pred == bool(y_train[n][1]))

    accuracy = float(sum(scores)) / len(predictions)

    print("Thresholding accuracy: {}%".format(accuracy))

def main():
    test_thresholding()


if __name__ == '__main__':
    main()
