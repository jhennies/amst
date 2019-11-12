
import os
import glob
import numpy as np
from multiprocessing import Pool
from tifffile import imread, imsave
import warnings


def median_z(target_folder, im_list, idx, radius=7):
    """
    Computes the z-median smoothed image slice at position idx

    :param target_folder: Where the result will be stored
    :param im_list: List of image file locations
    :param idx: The current position
    :param radius: Radius of the median smoothing
    :return:
    """

    filename = os.path.split(im_list[idx])[1]

    # Return if the target file already exists
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('median_z: {} exists, nothing to do'.format(filename))
        return

    print('median_z on slice {}'.format(idx))

    # Load necessary images
    if True:
        start_id = idx - radius
        # Cope with the boundaries
        if start_id < 0:
            start_id = 0
        end_id = idx + radius + 1
        if end_id > len(im_list):
            end_id = len(im_list)

        ims = []
        for load_id in range(start_id, end_id):
            # print('load_id={}'.format(load_id))
            ims.append(imread(im_list[load_id]))

    # Make composite
    composite = np.median(ims, axis=0).astype(ims[0].dtype)

    # Save result
    imsave(os.path.join(target_folder, filename), data=composite)


def parallel_median_z(source_folder, target_folder, radius=7, n_workers=1, source_range=np.s_[:]):
    """
    Wraps around a median_z for multiprocessing.

    :param source_folder: Where the source tif slices are
    :param target_folder: Location to store the results
    :param radius:
    :param n_workers: For multiprocessing
        n_workers == 1: normal function call
        n_workers > 1: multiprocessing
    :param source_range: for debugging to select a subset of the slices
    :return:
    """

    print('Computing {} with n_workers={}'.format(median_z, n_workers))
    print('radius = {}'.format(radius))

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[source_range]

    if n_workers == 1:

        for idx in range(len(im_list)):
            median_z(
                target_folder, im_list, idx, radius
            )

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    median_z, (
                        target_folder, im_list, idx, radius
                    )
                )
                for idx in range(len(im_list))
            ]

            [task.get() for task in tasks]
