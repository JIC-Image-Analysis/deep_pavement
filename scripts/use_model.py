
import click

from keras.models import load_model

from imageio import imread, imsave

import numpy as np


@click.command()
@click.argument('image_fpath')
def main(image_fpath):
    model = load_model('my_model.h5')

    image = imread(image_fpath)
    print(image.shape)
    flattened_image = image[:, :, 0]

    start_x, size_x = 500, 300
    start_y, size_y = 500, 300
    pad = 25
    interesting = image[start_x-pad:start_x+size_x+pad, start_y-pad:start_y+size_y+pad]
    imsave('interesting.png', interesting)
    dilated_image = imread('dilated_boundaries.png')
    dilated = dilated_image[start_x-pad:start_x+size_x+pad, start_y-pad:start_y+size_y+pad]
    imsave('ground_truth.png', dilated)

    # sections = []
    # pad = 25

    # for x in xrange(start_x, start_x+size_x):
    #     for y in xrange(start_y, start_y+size_y):
    #         section = flattened_image[x-pad:x+pad+1, y-pad:y+pad+1]
    #         sections.append(section)

    # stack = np.transpose(sections, (0, 1, 2))
    # n_images, xdim, ydim = stack.shape
    # to_classify = stack.reshape(n_images, xdim, ydim, 1)

    # predictions = model.predict(to_classify, batch_size=32)

    # reshaped = predictions.reshape(size_x, size_y, 2)

    # class_img = reshaped[:, :, 1]

    # imsave('classed.png', class_img)

if __name__ == '__main__':
    main()
