
import os
import numpy as np
import glob
from tifffile import imread, imsave
from scipy.ndimage.interpolation import shift
import csv
import pickle
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import pyplot as plt


def displace_slice(
        target_folder,
        im_filepath,
        displacement,
        subpx_displacement=False,
        compression=0,
        pad_zeros=None,
        bounds=np.s_[:],
        target_shape=None
):

    filename = os.path.split(im_filepath)[1]
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('_sift_align: {} exists, nothing to do'.format(filename))
        return

    print('displacements on {}'.format(filename))

    # displacement = np.array((
    #         displacement[0] - bounds[1].start,
    #         displacement[1] - bounds[0].start
    # ))

    # Load image
    im = imread(im_filepath)[bounds]
    if target_shape is not None:
        tim = np.zeros(target_shape, im.dtype)
        tim[:im.shape[0], :im.shape[1]] = im
        im = tim

    # zero-pad image
    if pad_zeros:
        pad_im = np.zeros(np.array(im.shape) + (2*pad_zeros), dtype=im.dtype)
        pad_im[pad_zeros: -pad_zeros, pad_zeros: -pad_zeros] = im
        im = pad_im

    print(f'displacement = {displacement}')

    # Shift the image
    if not subpx_displacement:
        im = shift(im, (np.round(displacement[1]), np.round(displacement[0])))
    else:
        im = shift(im, (displacement[1], displacement[0]))

    # Write result
    imsave(os.path.join(target_folder, filename), im.astype(im.dtype), compress=compression)


def displace_slices(
        im_list,
        target_folder,
        offsets,
        subpx_displacement=False,
        compression=0,
        pad_zeros=None,
        bounds=None,
        parallel_method='multi_process',
        n_workers=os.cpu_count()
):

    offsets = np.array(offsets)
    shape = None

    if bounds is not None:

        # Prepare the displacements respective all bounds and figure out the target shape
        offsets_ = []
        for idx, b in enumerate(bounds):
            offsets_.append(np.array((offsets[idx][0] + b[1].start, offsets[idx][1] + b[0].start)))
        offsets = offsets_
        offsets = offsets - np.min(offsets, axis=0)
        starts = []
        stops = []
        for idx, b in enumerate(bounds):
            starts.append(np.array((b[0].start, b[1].start)) + offsets[idx][::-1])
            stops.append(np.array((b[0].stop, b[1].stop)) + offsets[idx][::-1])
        min_yx = np.floor(np.min(starts, axis=0)).astype(int)
        max_yx = np.ceil(np.max(stops, axis=0)).astype(int)
        # Pad a little to each side to make it less squished
        # FIXME make this available as a parameter (amount of padding, e.g. pad_result)
        shape = max_yx - min_yx  # + 32
        # offsets += 16

    if n_workers == 1:

        print('Running with one worker...')
        for idx in range(len(im_list)):
            displace_slice(
                target_folder, im_list[idx], offsets[idx], subpx_displacement=subpx_displacement,
                compression=compression, pad_zeros=pad_zeros, bounds=bounds[idx] if bounds is not None else np.s_[:],
                target_shape=shape
            )

    else:

        if parallel_method == 'multi_process':
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:

                tasks = [
                    p.apply_async(
                        displace_slice, (
                            target_folder, im_list[idx], offsets[idx], subpx_displacement, compression, pad_zeros,
                            bounds[idx] if bounds is not None else np.s_[:], shape
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
                        displace_slice,
                        target_folder, im_list[idx], offsets[idx]
                    )
                    for idx in range(len(im_list))
                ]

                [task.result() for task in tasks]

        else:
            print("Only 'multi_process' and 'multi_thread' are implemented.")
            raise NotImplementedError


def smooth_offsets(offsets, median_radius=0, gaussian_sigma=0., suppress_x=False):

    disps_y = offsets[:, 1]
    if median_radius > 0:
        disps_y = medfilt(disps_y, median_radius * 2 + 1)
    if gaussian_sigma > 0:
        disps_y = gaussian_filter1d(disps_y, gaussian_sigma)

    if not suppress_x:

        disps_x = offsets[:, 0]
        if median_radius > 0:
            disps_x = medfilt(disps_x, median_radius * 2 + 1)
        if gaussian_sigma > 0:
            disps_x = gaussian_filter1d(disps_x, gaussian_sigma)

    else:
        disps_x = np.zeros(disps_y.shape, dtype=disps_y.dtype)

    return np.concatenate([disps_x[:, None], disps_y[:, None]], axis=1)


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
        pattern='*.tif',
        pad_zeros=None,
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

    def _displacements_from_fiji(disp_fp):

        # FIXME This currently assumes Fiji's 1-based slice numbering
        displacements = dict()
        with open(disp_fp) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    pass
                else:
                    displacements[int(row[0]) - 1] = (float(row[1]), float(row[2]))
                line_count += 1
        assert min(displacements.keys()) >= 0, "Slice indices are assumed in Fiji's 1-based slice numbering!"

        # Here we have to make sure that the reference slice is added to the displacements
        disps = np.array(
            [displacements[x] if x in displacements.keys() else (0., 0.) for x in range(len(displacements) + 1)])
        if verbose >= 2:
            plt.plot(disps)
        return disps

    def _displacements_from_pkl(disp_fp):
        with open(disp_fp, mode='rb') as f:
            disps = -np.array(pickle.load(f))
        # disps = np.concatenate([[[0, 0]], disps], axis=0)
        return disps

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, pattern))))[source_range]

    # Load the displacements
    if os.path.splitext(displacements_file)[1] == '.csv':
        disps_array = _displacements_from_fiji(displacements_file)
    elif os.path.splitext(displacements_file)[1] == '.pkl':
        disps_array = _displacements_from_pkl(displacements_file)
    else:
        raise ValueError(f'Invalid filetype: {os.path.splitext(displacements_file)[1]}')

    # Process the displacements
    disps_median = smooth_offsets(
        disps_array,
        median_radius=median_radius, gaussian_sigma=gaussian_sigma, suppress_x=suppress_x
    )

    if verbose >= 2:
        plt.figure()
        plt.plot(disps_median)

    displacements = disps_median

    displace_slices(
        im_list, target_folder, displacements,
        subpx_displacement=subpx_displacement,
        compression=compression,
        pad_zeros=pad_zeros,
        parallel_method=parallel_method,
        n_workers=n_workers
    )


def subtract_run_avg(offsets, sigma=10.):

    if type(sigma) != list:
        sigma = [sigma, sigma]

    offsets = np.array(offsets)
    offsets_x = offsets[:, 0]
    offsets_y = offsets[:, 1]
    if sigma[0] > 0:
        offsets_x = gaussian_filter1d(offsets_x, sigma[0])
    if sigma[1] > 0:
        offsets_y = gaussian_filter1d(offsets_y, sigma[1])

    running_average = np.concatenate([offsets_x[:, None], offsets_y[:, None]], axis=1)

    return offsets - running_average

