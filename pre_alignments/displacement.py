
import os
import numpy as np
import glob
from tifffile import imread, imsave
from scipy.ndimage.interpolation import shift
import csv
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import pyplot as plt


def displace(target_folder, im_filepath, displacement, subpx_displacement=False, compression=0):

    filename = os.path.split(im_filepath)[1]
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('_sift_align: {} exists, nothing to do'.format(filename))
        return

    print('displacements on {}'.format(filename))

    # Load image
    im = imread(im_filepath)

    # FIXME figure out if this is right
    # Shift the image
    if subpx_displacement:
        im = shift(im, (np.round(displacement[1]), np.round(displacement[0])))
    else:
        im = shift(im, (displacement[1], displacement[0]))

    # Write result
    imsave(os.path.join(target_folder, filename), im.astype(im.dtype), compress=compression)


def smooth_displace(
        source_folder, target_folder,
        displacements_file,
        median_radius=7,
        gaussian_sigma=7,
        n_workers=1, source_range=np.s_[:],
        parallel_method='multi_process',
        suppress_x=False,
        subpx_displacement=False,
        compression=0,
        verbose=0
):
    """
    Load a displacement file, smooth the displacements first by median and second by Gaussian smoothing and apply the
    displacements to a dataset.

    :param source_folder: The folder from which to take the moving images
    :param target_folder: The folder to store the results
    :param displacements_file: File where displacements are stored ('save as' from Fiji's results panel)
    :param n_workers: For multiprocessing
        n_workers == 1: normal function call
        n_workers > 1: multiprocessing
    :param source_range: for debugging to select a subset of the moving images
    :param parallel_method: Either 'multi_process' or 'multi_thread'
    :return:
    """

    print('Computing {} with n_workers={}'.format(displace, n_workers))

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[source_range]

    # Load the displacements
    # FIXME This currently assumes Fiji's 1-based slice numbering
    displacements = dict()
    with open(displacements_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                displacements[int(row[0]) - 1] = (float(row[1]), float(row[2]))
            line_count += 1
    assert min(displacements.keys()) >= 0, "Slice indices are assumed in Fiji's 1-based slice numbering!"

    # Process the displacements
    disps_array = np.array([displacements[x] if x in displacements.keys() else (0., 0.) for x in range(max(displacements.keys()) + 1)])
    if verbose >= 2:
        plt.plot(disps_array)

    disps_y = disps_array[:, 1]
    if median_radius > 0:
        disps_y = medfilt(disps_y, median_radius * 2 + 1)
    if gaussian_sigma > 0:
        disps_y = gaussian_filter1d(disps_y, gaussian_sigma)

    if not suppress_x:

        disps_x = disps_array[:, 0]
        if median_radius > 0:
            disps_x = medfilt(disps_x, median_radius * 2 + 1)
        if gaussian_sigma > 0:
            disps_x = gaussian_filter1d(disps_x, gaussian_sigma)

    else:
        disps_x = np.zeros(disps_y.shape, dtype=disps_y.dtype)

    disps_median = np.concatenate([disps_x[:, None], disps_y[:, None]], axis=1)
    if verbose >= 2:
        plt.figure()
        plt.plot(disps_median)

    displacements = disps_median

    if n_workers == 1:

        print('Running with one worker...')
        for idx in range(len(im_list)):
            displace(
                target_folder, im_list[idx], displacements[idx], subpx_displacement=subpx_displacement, compression=compression
            )

    else:

        if parallel_method == 'multi_process':
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:

                tasks = [
                    p.apply_async(
                        displace, (
                            target_folder, im_list[idx], displacements[idx], subpx_displacement, compression
                        )
                    )
                    for idx in range(len(im_list))
                ]

                [task.get() for task in tasks]

        elif parallel_method == 'multi_thread':
            from concurrent.futures import ThreadPoolExecutor as TPool
            with TPool(max_workers=n_workers) as p:

                tasks = [
                    p.submit(
                        displace,
                        target_folder, im_list[idx], displacements[idx]
                    )
                    for idx in range(len(im_list))
                ]

                [task.result() for task in tasks]

        else:
            print("Only 'multi_process' and 'multi_thread' are implemented.")
            raise NotImplementedError

