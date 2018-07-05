import os
import errno
import random

import numpy as np

import imageio

from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

import click

DILATION_SIZE = 2


def load_segmentation_from_rgb_image(filename):

    rgb_image = imageio.imread(filename)

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

    imageio.imsave('dilated_boundaries.png', 255 * dilated_boundaries)

    return dilated_boundaries


def generate_merged_image(projection, dilated_boundaries):
    merged_image = np.copy(projection)

    merged_image[np.where(dilated_boundaries)] = [255, 0, 0]

    imageio.imsave('merged_image.png', merged_image)


def create_and_save_sample_images(
    coords_list,
    image,
    base_path,
    sample_size=100,
    pad=25
):

    short_coords_list = random.sample(coords_list, 2*sample_size)
    coords_iterator = iter(short_coords_list)

    try:
        os.makedirs(base_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    def make_section(coords):
        x, y = coords
        section = image[x-pad:x+pad+1, y-pad:y+pad+1]
        return section

    expected_shape = (2 * pad + 1, 2 * pad + 1)
    n = 0
    while n < sample_size:
        coords = next(coords_iterator)
        section = make_section(coords)
        if section.shape == expected_shape:
            output_fpath = os.path.join(base_path, "section{}.png".format(n))
            imageio.imsave(output_fpath, section)
            n = n + 1


def mangle_image(segmentation_fpath, projection_fpath):

    projection = imageio.imread(projection_fpath)
    flattened_projection = projection[:, :, 0]

    dilated_boundaries = generate_dilated_boundary_image(segmentation_fpath)

    cell_wall_arrays = np.where(dilated_boundaries)
    cell_wall_coordinates = zip(*cell_wall_arrays)

    test_size = 1000
    train_size = 10000

    create_and_save_sample_images(
        cell_wall_coordinates,
        flattened_projection,
        "data/train/wall",
        sample_size=train_size
    )
    create_and_save_sample_images(
        cell_wall_coordinates,
        flattened_projection,
        "data/test/wall",
        sample_size=test_size
    )

    not_wall_arrays = np.where(dilated_boundaries == 0)
    not_wall_coordinates = zip(*not_wall_arrays)

    create_and_save_sample_images(
        not_wall_coordinates,
        flattened_projection,
        "data/train/not_wall",
        sample_size=train_size
    )
    create_and_save_sample_images(
        not_wall_coordinates,
        flattened_projection,
        "data/test/not_wall",
        sample_size=test_size
    )


@click.command()
@click.argument('segmentation_fpath')
@click.argument('projection_fpath')
def main(segmentation_fpath, projection_fpath):

    mangle_image(segmentation_fpath, projection_fpath)


if __name__ == '__main__':
    main()
