
import click

from keras.models import load_model

from imageio import imread, imsave

import numpy as np


@click.command()
@click.argument('image_fpath')
def main(image_fpath):
    model = load_model('unet256.h5')

    image = imread(image_fpath)
    print(image.shape)

    results = model.predict(image)

    print(results.shape)
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
