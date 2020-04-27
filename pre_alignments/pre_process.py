
from tifffile import imsave, imread
from glob import glob
import numpy as np
import os
from multiprocessing import Pool


def _crop(x, y, w, h, source_filepath, target_folder, compression, verbose=0):

    target_filepath = os.path.join(
        target_folder,
        os.path.split(source_filepath)[1]
    )
    if verbose:
        print(target_filepath)

    im = imread(source_filepath)
    im = im[y: y + h, x: x + w]
    imsave(target_filepath, data=im, compress=compression)


def crop_fiji_region(
        x, y, w, h,
        source_folder,
        target_folder,
        pattern='*.tif',
        compression=0,
        z_range=np.s_[:],
        n_workers=1,
        verbose=0
):

    assert source_folder != target_folder, 'The results folder must differ from the input folder!'

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    im_list = sorted(glob(os.path.join(source_folder, pattern)))[z_range]
    if verbose:
        print(im_list)

    if n_workers == 1:

        [_crop(x, y, w, h, source, target_folder, compression, verbose) for source in im_list]

    else:

        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(
                    _crop, (x, y, w, h, source, target_folder, compression, verbose)
                )
                for source in im_list
            ]
            [task.get() for task in tasks]
