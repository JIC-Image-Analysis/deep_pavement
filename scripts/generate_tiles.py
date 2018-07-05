from pathlib import Path

import click

import numpy as np

from imageio import imread, imsave
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

from dtoolcore import DataSet


DILATION_SIZE = 1


def load_segmentation_from_rgb_image(filename):

    rgb_image = imread(filename)

    ydim, xdim, _ = rgb_image.shape

    segmentation = np.zeros((ydim, xdim), dtype=np.uint32)

    segmentation += rgb_image[:, :, 2]
    segmentation += rgb_image[:, :, 1].astype(np.uint32) * 256
    segmentation += rgb_image[:, :, 0].astype(np.uint32) * 256 * 256

    return segmentation


def generate_dilated_boundary_image(image_fpath):

    segmentation = load_segmentation_from_rgb_image(image_fpath)

    boundaries = find_boundaries(segmentation)

    dilated_boundaries = dilation(boundaries, disk(DILATION_SIZE))

    return 255 * dilated_boundaries.astype(np.uint8)


def generate_tiles(dataset, output_path):


    projection_id = 'a6b413939aea04bd96ca03b8cb7a98294c65f0fd'
    segmentation_id = '963efa345525bce11bbb00663fcf838cb0f37c9a'

    projection_fpath = dataset.item_content_abspath(projection_id)
    segmenatation_fpath = dataset.item_content_abspath(segmentation_id)

    projection_image = imread(projection_fpath)

    ts = 256

    sx, sy = 1200, 1000

    tile = projection_image[sx:sx+ts,sy:sy+ts,:]

    imsave('tile.png', tile)

    dilated_boundaries = generate_dilated_boundary_image(segmenatation_fpath)

    mask_tile = dilated_boundaries[sx:sx+ts,sy:sy+ts]
    imsave('mask_tile.png', mask_tile)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    dataset = DataSet.from_uri(dataset_uri)

    generate_tiles(dataset, 'tile_data')


if __name__ == '__main__':
    main()
