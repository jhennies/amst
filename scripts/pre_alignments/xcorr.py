
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


def _xcorr(image, offset_image, thresh=0, sigma=1.):
    if type(thresh) != list:
        thresh = [thresh, thresh]
    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
        offset_image[offset_image < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]
        offset_image[offset_image > thresh[1]] = thresh[1]
    image = gaussianSmoothing(image, sigma)
    offset_image = gaussianSmoothing(offset_image, sigma)
    image = filters.sobel(image)
    offset_image = filters.sobel(offset_image)
    shift, error, diffphase = register_translation(image, offset_image, 10)
    return shift[1], shift[0]


def _displace_slice(image_filepath, offset, result_filepath=None, subpx_displacement=False, compression=0):
    print('Writing {}'.format(os.path.split(image_filepath)[1]))
    image = imread(image_filepath)
    if subpx_displacement:
        image = shift(image, [-offset[1], -offset[0]])
    else:
        image = shift(image, -np.round([offset[1], offset[0]]))

    if result_filepath is not None:
        imsave(result_filepath, image, compress=compression)

    return image


def _wrap_xcorr(im_idx, im_list, xy_range, thresh=0, sigma=1.):
    print('{} / {}'.format(im_idx + 1, len(im_list)))

    im_filepath = im_list[im_idx]
    im_ref_filepath = im_list[im_idx - 1]

    print('moving: {}'.format(os.path.split(im_filepath)[1]))
    print('fixed: {}'.format(os.path.split(im_ref_filepath)[1]))

    offset = _xcorr(imread(im_filepath)[xy_range], imread(im_ref_filepath)[xy_range], thresh=thresh, sigma=sigma)
    return offset


def _sequentialize_offsets(offsets):

    seq_offset = np.array([0., 0.], dtype='float32')
    seq_offsets = [seq_offset.copy()]

    for offset in offsets:
        seq_offset += np.array(offset)
        seq_offsets.append(seq_offset.copy())

    return seq_offsets


def _subtract_running_average(offsets, sigma=10.):

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


def offsets_with_xcorr(
        source_folder, target_folder=None,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        subtract_running_average=0,
        subpixel_displacements=False,
        threshold=0,
        sigma=1.,
        compression=0,
        return_sequential=False,
        n_workers=1,
        verbose=0
):

    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    im_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))[z_range]

    if n_workers == 1:
        offsets = []
        for im_idx in range(1, len(im_list)):
            offsets.append(_wrap_xcorr(im_idx, im_list, xy_range, thresh=threshold, sigma=sigma))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_wrap_xcorr, im_idx, im_list, xy_range, threshold, sigma)
                for im_idx in range(1, len(im_list))
            ]
        offsets = [task.result() for task in tasks]

    if target_folder is not None or return_sequential:

        print('Sequentializing offsets ...')
        seq_offsets = _sequentialize_offsets(offsets)

        if verbose:
            plt.plot(offsets)
            plt.figure()
            plt.plot(seq_offsets)

        if subtract_running_average:
            print('Subtracting running average ...')
            seq_offsets = _subtract_running_average(seq_offsets, subtract_running_average)

        if verbose:
            plt.figure()
            plt.plot(seq_offsets)
            plt.show()

    if target_folder is not None:

        print('Displacing and saving slices ...')

        if n_workers == 1:
            for im_idx, im_filepath in enumerate(im_list):
                result_filepath = os.path.join(target_folder, os.path.split(im_filepath)[1])
                if not os.path.exists(result_filepath):
                    _displace_slice(im_filepath,
                                    seq_offsets[im_idx],
                                    result_filepath=result_filepath,
                                    subpx_displacement=subpixel_displacements,
                                    compression=compression)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as tpe:
                tasks = [
                    tpe.submit(_displace_slice,
                               im_filepath,
                               seq_offsets[im_idx],
                               os.path.join(target_folder, os.path.split(im_filepath)[1]),
                               subpixel_displacements,
                               compression)
                    for im_idx, im_filepath in enumerate(im_list)
                    if not os.path.exists(os.path.join(target_folder, os.path.split(im_filepath)[1]))
                ]
            [task.result() for task in tasks]

    if return_sequential:
        return seq_offsets
    else:
        return offsets


if __name__ == '__main__':

    source = '/data/tmp/alignment_for_wioleta/subset_1001'
    target = '/data/tmp/alignment_for_wioleta/xcorr_align_2'

    # z_range = np.s_[320:2100]
    z_range = np.s_[800:900]
    # z_range = np.s_[:]

    offsets_with_xcorr(source, target, z_range=z_range, xy_range=np.s_[2780:3392, 2370:4290], n_workers=12)
