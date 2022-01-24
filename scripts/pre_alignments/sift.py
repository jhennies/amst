
from skimage.feature import register_translation
from vigra.filters import gaussianSmoothing, gaussianGradientMagnitude
from skimage import filters
import glob
import os
from tifffile import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter1d
from silx.image import sift

from .displacement import displace_slices, subtract_run_avg
from .slice_pre_processing import preprocess_slice
from .data_generation import parallel_image_slice_generator


def _sift(image, reference, sift_ocl=None, devicetype=None, return_keypoints=False, verbose=False):

    if sift_ocl is None and devicetype is None:
        raise RuntimeError('Either sift_ocl or devicetype need to be supplied')

    # Initialize the SIFT
    if sift_ocl is None:
        if verbose:
            print('Initializing SIFT')
        assert devicetype is not None
        sift_ocl = sift.SiftPlan(template=image, devicetype=devicetype)

    if verbose:
        print('Computing keypoints')

    # Compute keypoints
    keypoints_moving = sift_ocl(image)
    if verbose:
        print(f'type(reference) = {type(reference)}')
    if type(reference) == np.ndarray:
        keypoints_ref = sift_ocl(reference)
    else:
        if reference is None:
            return (0., 0.), keypoints_moving
        keypoints_ref = reference

    if verbose:
        print('Matching keypoints')

    # Match keypoints
    mp = sift.MatchPlan()
    match = mp(keypoints_ref, keypoints_moving)

    if verbose:
        print('Computing offset')

    # Determine offset
    if len(match) == 0:
        print('Warning: No matching keypoints found!')
        offset = (0., 0.)
    else:
        offset = (np.median(match[:, 1].x - match[:, 0].x), np.median(match[:, 1].y - match[:, 0].y))

    if return_keypoints:
        return (offset[0], offset[1]), keypoints_moving
    else:
        return offset[0], offset[1]


def _init_sift(im_filepath, devicetype='GPU'):

    # Initialize the SIFT
    sift_ocl = sift.SiftPlan(template=imread(im_filepath), devicetype=devicetype)

    return sift_ocl


def _wrap_sift(
        im_list, xy_range,
        thresh=(0, 0), sigma=1., mask_range=None, devicetype='GPU',
        n_workers=os.cpu_count(),
        n_gpus=1,
        verbose=False
):
    assert n_gpus == 1, 'Only implemented for the use of one GPU'

    if verbose:
        print(f'devicetype = {devicetype}')
        print(f'n_workers = {n_workers}')
        print(f'thresh = {thresh}')
        print(f'sigma = {sigma}')
        print(f'mask_range = {mask_range}')

    # slice_gen = parallel_image_slice_generator(
    #     im_list, xy_range,
    #     preprocess_slice, {'thresh': thresh, 'sigma': sigma, 'mask_range': mask_range},
    #     yield_consecutive=True,
    #     n_workers=n_workers
    # )
    #
    # return [
    #     _sift(ims[1], ims[0], devicetype=devicetype, verbose=verbose)
    #     for ims in slice_gen
    # ]

    slice_gen = parallel_image_slice_generator(
        im_list, xy_range,
        preprocess_slice, {'thresh': thresh, 'sigma': sigma, 'mask_range': mask_range},
        yield_consecutive=False,
        n_workers=n_workers
    )

    keypoints = None
    sift_ocl = _init_sift(im_list[0], devicetype=devicetype)
    offsets = []
    for idx, im in enumerate(slice_gen):
        print(f'SIFT on image {idx}: {im_list[idx]}')
        offset, keypoints = _sift(
            im, keypoints,
            sift_ocl=sift_ocl,
            return_keypoints=True, verbose=verbose
        )
        offsets.append(offset)

    return offsets[1:]  # Do not return the offset of the first slice (which is 0,0)


def _sequentialize_offsets(offsets):

    seq_offset = np.array([0., 0.], dtype='float32')
    seq_offsets = [seq_offset.copy()]

    for offset in offsets:
        seq_offset += np.array(offset)
        seq_offsets.append(seq_offset.copy())

    return seq_offsets


def offsets_with_sift(
        source_folder, target_folder=None,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        subtract_running_average=0,
        subpixel_displacement=False,
        threshold=(0, 0),
        mask_range=None,
        sigma=1.,
        compression=0,
        return_sequential=False,
        devicetype='GPU',
        n_workers=os.cpu_count(),
        n_gpus=1,
        verbose=False
):

    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    if mask_range is not None:
        if mask_range[1] < 0:
            mask_range[1] = 256 + mask_range[1]

    im_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))[z_range]

    offsets = _wrap_sift(
        im_list, xy_range, thresh=threshold, sigma=sigma, mask_range=mask_range,
        n_workers=n_workers,
        n_gpus=n_gpus,
        devicetype=devicetype,
        verbose=verbose
    )

    offsets = -np.array(offsets)

    if target_folder is not None or return_sequential:

        print('Sequentializing offsets ...')
        seq_offsets = _sequentialize_offsets(offsets)

        if verbose:
            plt.plot(offsets)
            plt.figure()
            plt.plot(seq_offsets)

        if subtract_running_average:
            print('Subtracting running average ...')
            seq_offsets = subtract_run_avg(seq_offsets, subtract_running_average)

            if verbose:
                plt.figure()
                plt.plot(seq_offsets)
                plt.show()

        if target_folder is not None:

            print('Displacing and saving slices ...')

            displace_slices(
                im_list, target_folder, seq_offsets,
                subpx_displacement=subpixel_displacement,
                compression=compression,
                pad_zeros=None,
                parallel_method='multi_process',
                n_workers=n_workers
            )

        if return_sequential:
            return seq_offsets
    return offsets


if __name__ == '__main__':
    pass
