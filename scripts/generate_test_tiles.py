
from pathlib import Path

import numpy as np
from imageio import imread, imsave

import click

@click.command()
@click.argument('image_fpath')
@click.argument('output_path')
def main(image_fpath, output_path):

    output_path = Path(output_path)

    image = imread(image_fpath)

    xdim = image.shape[0]
    ydim = image.shape[1]

    ts = 256

    nx = xdim//ts
    ny = ydim//ts


    for x in range(nx):
        for y in range(ny):

            tile = image[x*ts:(x+1)*ts,y*ts:(y+1)*ts]
            tile = np.dstack([tile] * 3)
            imsave(output_path/'test_tile-{}-{}.png'.format(x, y), tile)


if __name__ == '__main__':
    main()
