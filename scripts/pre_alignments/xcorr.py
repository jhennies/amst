
from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
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
from .displacement import displace_slices, subtract_run_avg


def _norm_8bit(im):
    im = im.astype('float32')
    upper = np.quantile(im, 0.8)
    lower = np.quantile(im, 0.2)
    im -= lower
    im /= (upper - lower)
    im *= 255
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def _xcorr(image, offset_image, thresh=(0, 0), sigma=1., mask_range=None):
    if mask_range is not None:
        image[image < mask_range[0]] = 0
        image[image > mask_range[1]] = 0
    if type(thresh) != list:
        thresh = [thresh, thresh]
    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
        offset_image[offset_image < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]
        offset_image[offset_image > thresh[1]] = thresh[1]

    # dtype = image.dtype

    image = _norm_8bit(image)
    offset_image = _norm_8bit(offset_image)

    # image = gaussianSmoothing(image.astype('float32'), sigma).astype(dtype)
    # offset_image = gaussianSmoothing(offset_image.astype('float32'), sigma).astype(dtype)
    image = gaussianSmoothing(image, sigma)
    offset_image = gaussianSmoothing(offset_image, sigma)
    image = filters.sobel(image)
    offset_image = filters.sobel(offset_image)
    # shift, error, diffphase = register_translation(image, offset_image, 10)
    shift, error, diffphase = phase_cross_correlation(image, offset_image, upsample_factor=10)
    return shift[1], shift[0]


def _wrap_xcorr(im_idx, im_list, xy_range, thresh=(0, 0), sigma=1., mask_range=None, num_ref_ims=1):
    print('{} / {}'.format(im_idx + 1, len(im_list)))

    im_filepath = im_list[im_idx]
    im = imread(im_filepath)[xy_range]
    if num_ref_ims == 1:
        im_ref_filepath = im_list[im_idx - 1]
        im_ref = imread(im_ref_filepath)[xy_range]
    else:
        im_ref = []
        for idx in range(num_ref_ims):
            if im_idx - 1 - idx >= 0:
                im_ref.append(imread(im_list[im_idx - 1 - idx])[xy_range])
        im_ref = np.mean(im_ref, axis=0)

    # print('moving: {}'.format(os.path.split(im_filepath)[1]))
    # print('fixed: {}'.format(os.path.split(im_ref_filepath)[1]))

    offset = _xcorr(
        im, im_ref,
        thresh=thresh, sigma=sigma, mask_range=mask_range
    )
    return offset


def _sequentialize_offsets(offsets):

    seq_offset = np.array([0., 0.], dtype='float32')
    seq_offsets = [seq_offset.copy()]

    for offset in offsets:
        seq_offset += np.array(offset)
        seq_offsets.append(seq_offset.copy())

    return seq_offsets


def offsets_with_xcorr(
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
        local_ref_slices=1,
        n_workers=1,
        verbose=0
):

    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    if mask_range is not None:
        if mask_range[1] < 0:
            mask_range[1] = 256 + mask_range[1]

    im_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))[z_range]

    if n_workers == 1:
        offsets = []
        for im_idx in range(1, len(im_list)):
            offsets.append(_wrap_xcorr(
                im_idx, im_list, xy_range, thresh=threshold, sigma=sigma, mask_range=mask_range, num_ref_ims=local_ref_slices
            ))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_wrap_xcorr, im_idx, im_list, xy_range, threshold, sigma, mask_range, local_ref_slices)
                for im_idx in range(1, len(im_list))
            ]
            offsets = [task.result() for task in tasks]

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

    source = '/data/tmp/alignment_for_wioleta/subset_1001'
    target = '/data/tmp/alignment_for_wioleta/xcorr_align_2'

    # z_range = np.s_[320:2100]
    z_range = np.s_[800:900]
    # z_range = np.s_[:]

    offsets_with_xcorr(source, target, z_range=z_range, xy_range=np.s_[2780:3392, 2370:4290], n_workers=12)
