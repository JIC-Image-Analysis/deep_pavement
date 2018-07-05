import os

import numpy as np

import keras
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


from imageio import imread, imsave


def load_stuff():

    base_path = 'test_pos'
    images = []
    for n in range(100):
        fpath = os.path.join(base_path, 'section{}.png'.format(n))
        images.append(imread(fpath))
    base_path = 'test_neg'
    for n in range(100):
        fpath = os.path.join(base_path, 'section{}.png'.format(n))
        images.append(imread(fpath))
    stack = np.transpose(images, (0, 1, 2))
    n_images, xdim, ydim = stack.shape
    x_test = stack.reshape(n_images, xdim, ydim, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = [1] * 100 + [0] * 100
    y_test = np_utils.to_categorical(y_test)

    train_size = 100
    # Channels last
    base_path = 'positive_examples'
    images = []
    for n in range(train_size):
        fpath = os.path.join(base_path, 'section{}.png'.format(n))
        images.append(imread(fpath))
    base_path = 'negative_examples'
    for n in range(train_size):
        fpath = os.path.join(base_path, 'section{}.png'.format(n))
        images.append(imread(fpath))
    stack = np.transpose(images, (0, 1, 2))

    n_images, xdim, ydim = stack.shape

    # Channels last
    x_train = stack.reshape(n_images, xdim, ydim, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = [1] * train_size + [0] * train_size

    y_train = np_utils.to_categorical(y_train)
    print(y_train)
    sys.exit(0)

    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(51, 51, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('my_model.h5')


def main():
    load_stuff()


if __name__ == '__main__':
    main()
