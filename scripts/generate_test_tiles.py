
from pathlib import Path

import numpy as np
from imageio import imread, imsave

import click

@click.command()
@click.argument('image_fpath')
@click.argument('output_path')
def main(image_fpath, output_path):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image = imread(image_fpath)

    xdim = image.shape[0]
    ydim = image.shape[1]

    ts = 256

    nx = xdim//ts
    ny = ydim//ts

    pad_x = ts * (nx + 1) - xdim
    pad_y = ts * (ny + 1) - ydim

    padded_image = np.pad(image, ((0, pad_x), (0, pad_y), (0, 0)), 'constant', constant_values=(1, 1))

    print('Padding to {}'.format(padded_image.shape))
    for x in range(nx+1):
        for y in range(ny+1):
            tile = padded_image[x*ts:(x+1)*ts,y*ts:(y+1)*ts]
            # tile = np.dstack([tile] * 3)
            imsave(output_path/'test_tile-{}-{}.png'.format(x, y), tile)


if __name__ == '__main__':
    main()
