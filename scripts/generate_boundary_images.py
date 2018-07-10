from pathlib import Path
from collections import defaultdict

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


def generate_tiles_for_data_point(
    projection_fpath,
    segmentation_fpath,
    image_output,
    mask_output,
    ts=256
):

    projection_image = imread(projection_fpath)
    dilated_boundaries = generate_dilated_boundary_image(segmentation_fpath)

    nt = projection_fpath[-6:-4]

    # One image is grayscale :/
    try:
        xdim, ydim, _ = projection_image.shape
    except ValueError:
        return

    nx = xdim//ts
    ny = ydim//ts

    for x in range(nx):
        for y in range(ny):
            tile = projection_image[x*ts:(x+1)*ts,y*ts:(y+1)*ts]
            if tile.mean() > 10:
                imsave(image_output/'tile-{}-{}-{}.png'.format(nt, x, y), tile)
                mask = dilated_boundaries[x*ts:(x+1)*ts,y*ts:(y+1)*ts]
                imsave(mask_output/'tile-{}-{}-{}.png'.format(nt, x, y), mask)


def generate_boundary_image(dp_label, segmentation_fpath):

    output_path = Path('boundaries')
    output_path.mkdir(exist_ok=True)

    dilated_boundary_image = generate_dilated_boundary_image(segmentation_fpath)

    imsave(output_path/'{}-boundaries.png'.format(dp_label), dilated_boundary_image)


def generate_all_boundaries(dataset, output_path):

    # TODO - generate only where mask has border pixels

    image_output = output_path/'image'
    mask_output = output_path/'mask'
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    image_types = defaultdict(dict)
    for i in dataset.identifiers:
        relpath = dataset.item_properties(i)['relpath']
        datapoint_label = relpath[-7:-4]
        image_type = relpath[:-9]
        image_types[datapoint_label][image_type] = i

    for dp_label, ids in image_types.items():
        projection_id = ids['projection']
        segmentation_id = ids['segmentation']
        projection_fpath = dataset.item_content_abspath(projection_id)
        segmentation_fpath = dataset.item_content_abspath(segmentation_id)
        generate_boundary_image(dp_label, segmentation_fpath)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    dataset = DataSet.from_uri(dataset_uri)

    generate_all_boundaries(dataset, Path('tile_data'))


if __name__ == '__main__':
    main()
