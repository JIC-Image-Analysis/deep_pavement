import os

import click
import dtoolcore

import numpy as np

from imageio import imread, imsave

from dtoolbioimage import Image
from dtoolbioimage.segment import Segmentation
from dtool_utils.derived_dataset import DerivedDataSet

from parameters import Parameters

DILATION_SIZE = 2

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk


def generate_dilated_boundary_image(segmentation):

    boundaries = find_boundaries(segmentation)

    dilated_boundaries = dilation(boundaries, disk(DILATION_SIZE))

    # dilated_boundaries.view(Image).save('dilated_boundaries.png')

    return dilated_boundaries


def iter_idn_relpath(ds):

    for idn in ds.identifiers:
        yield idn, ds.item_properties(idn)["relpath"]


def generate_image_and_mask(seg, proj):

    mask = generate_dilated_boundary_image(seg)

    if len(proj.shape) == 2:
        proj = np.dstack(3 * [proj])

    proj[np.where(seg==0)] = 0

    return proj, mask


def images_and_masks_to_dataset(im_mask_generator, output_ds):

    for n, (image, mask) in enumerate(im_mask_generator):
        image_relpath = 'images/image{:02d}.png'.format(n)
        mask_relpath = 'masks/mask{:02d}.png'.format(n)
        image_fpath = output_ds.staging_fpath(image_relpath)
        mask_fpath = output_ds.staging_fpath(mask_relpath)

        mask.view(Image).save(mask_fpath)
        image.view(Image).save(image_fpath)

        mask_id = dtoolcore.utils.generate_identifier(mask_relpath)
        output_ds.add_item_metadata(image_relpath, "mask_ids", mask_id)
        output_ds.add_item_metadata(mask_relpath, "mask_ids", None)
        output_ds.add_item_metadata(image_relpath, "is_image", True)
        output_ds.add_item_metadata(mask_relpath, "is_image", False)


def create_image_mask_generator(input_ds, params, relpath_filter, seg_to_proj):

    segmentation_files = [
        (idn, relpath)
        for idn, relpath in iter_idn_relpath(input_ds)
        if relpath_filter(relpath)
    ]

    projection_files = [
        seg_to_proj(relpath)
        for idn, relpath in segmentation_files
    ]

    segs_projs = (
        (
            Segmentation.from_file(input_ds.item_content_abspath(seg_idn)),
            imread(input_ds.item_content_abspath(proj_idn))
        )
        for (seg_idn, _), (proj_idn, _) in zip(segmentation_files, projection_files)
    )

    return (generate_image_and_mask(seg, proj) for seg, proj in segs_projs)


def get_images_and_masks(input_ds, params):

    def relpath_filter(relpath):
        return relpath.startswith('segmentations')

    def get_proj_idn_relpath_from_segmentation(seg_relpath):
        basename = seg_relpath.split('/')[-1]
        # NOTE - basename[4:] works for files likes Seg_something_T03.png
        proj_relpath = 'projections/' + basename[4:]
        proj_idn = dtoolcore.utils.generate_identifier(proj_relpath)

        if proj_idn not in input_ds.identifiers:
            raise Exception("Can't find idn for {}".format(proj_relpath))

        return proj_idn, proj_relpath

    def da1_relpath_filter(relpath):
        return 'ws_seg' in relpath

    stem = "ws_seg.png"
    def da1_get_proj_idn_relpath_from_segmentation(seg_relpath):
        proj_relpath = seg_relpath[:-len(stem)]
        proj_idn = dtoolcore.utils.generate_identifier(proj_relpath)
        assert proj_idn in input_ds.identifiers

        return proj_idn, proj_relpath
        
    return create_image_mask_generator(
        input_ds,
        params,
        da1_relpath_filter,
        da1_get_proj_idn_relpath_from_segmentation
    )


def generate_image_mask_dataset(input_ds, output_ds, params):

    image_mask_generator = get_images_and_masks(input_ds, params)
    images_and_masks_to_dataset(image_mask_generator, output_ds)


@click.command()
@click.argument('input_ds_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(input_ds_uri, output_base_uri, output_name):

    input_ds = dtoolcore.DataSet.from_uri(input_ds_uri)

    params = Parameters()
    params['dilation_size'] = 2

    with DerivedDataSet(output_base_uri, output_name, input_ds) as output_ds:
        output_ds.readme_dict['parameters'] = params.parameter_dict
        generate_image_mask_dataset(input_ds, output_ds, params)


if __name__ == "__main__":
    main()
