import re

from pathlib import Path

import numpy as np

import click

from imageio import imread, imsave


def grouper(n, iterable):
    args = [iter(iterable)] * n
    return map(list, zip(*args))

def sorted_nicely(l):
    """Return list sorted in the way that humans expect.

    :param l: iterable to be sorted
    :returns: sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    sort_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=sort_key)

def join_tiles_in_path(path, output_fpath='joined.png'):

    files = sorted_nicely([str(f) for f in path.iterdir() if f.is_file()])

    n_rows = 1 + max([int(fn.split('-')[3]) for fn in files])

    images = [imread(f) for f in files]

    joined = np.block(list(grouper(n_rows, images)))

    print(joined.shape)

    imsave(output_fpath, joined)


@click.command()
@click.argument('tile_dir')
def main(tile_dir):

    tile_dir = Path(tile_dir)

    join_tiles_in_path(tile_dir)


if __name__ == '__main__':
    main()
