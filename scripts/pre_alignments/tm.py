
from skimage.feature import match_template
from vigra.filters import gaussianSmoothing, gaussianGradientMagnitude
from skimage import filters
import glob
import os
from tifffile import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .xcorr import _displace_slice


def _tm(image, template_im, thresh=0, sigma=1.):

    # print(f'template_im.shape = {template_im.shape}')
    # print(f'image.shape = {image.shape}')

    if type(thresh) != list:
        thresh = [thresh, thresh]
    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
        template_im[template_im < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]
        template_im[template_im > thresh[1]] = thresh[1]
    if sigma > 0:
        image = gaussianSmoothing(image, sigma)
        template_im = gaussianSmoothing(template_im, sigma)

    # print(f'template_im.shape = {template_im.shape}')
    # print(f'image.shape = {image.shape}')

    result = match_template(image, template_im)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    return x, y


def _wrap_tm(im_idx, im_list, im_ref, xy_range, thresh=0, sigma=1.):
    print('{} / {}'.format(im_idx + 1, len(im_list)))

    im_filepath = im_list[im_idx]

    print('moving: {}'.format(os.path.split(im_filepath)[1]))

    offset = _tm(imread(im_filepath)[xy_range], imread(im_ref), thresh=thresh, sigma=sigma)
    return offset


def offsets_with_tm(
        source_folder,
        ref_im,
        target_folder=None,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        subpixel_displacements=False,
        threshold=0,
        sigma=1.,
        add_offset=None,
        compression=0,
        n_workers=1,
        verbose=False
):

    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    im_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))[z_range]

    if n_workers == 1:
        offsets = []
        for im_idx in range(len(im_list)):
            offsets.append(_wrap_tm(im_idx, im_list, ref_im, xy_range, threshold, sigma))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_wrap_tm, im_idx, im_list, ref_im, xy_range, threshold, sigma)
                for im_idx in range(len(im_list))
            ]
            offsets = [task.result() for task in tasks]

    if add_offset is not None:
        print(offsets)
        print(f'Adapting offsets with {add_offset}')
        offsets = [off - np.array(add_offset) for off in offsets]
        print(offsets)

    if target_folder is not None:

        print('Displacing and saving slices ...')

        if n_workers == 1:
            for im_idx, im_filepath in enumerate(im_list):
                result_filepath = os.path.join(target_folder, os.path.split(im_filepath)[1])
                if not os.path.exists(result_filepath):
                    _displace_slice(im_filepath,
                                    offsets[im_idx],
                                    result_filepath=result_filepath,
                                    subpx_displacement=subpixel_displacements,
                                    compression=compression)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as tpe:
                tasks = [
                    tpe.submit(_displace_slice,
                               im_filepath,
                               offsets[im_idx],
                               os.path.join(target_folder, os.path.split(im_filepath)[1]),
                               subpixel_displacements,
                               compression)
                    for im_idx, im_filepath in enumerate(im_list)
                    if not os.path.exists(os.path.join(target_folder, os.path.split(im_filepath)[1]))
                ]
                [task.result() for task in tasks]

    return offsets


if __name__ == '__main__':

    pass
