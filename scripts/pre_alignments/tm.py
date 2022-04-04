
from skimage.feature import match_template
from vigra.filters import gaussianSmoothing, gaussianGradientMagnitude
from skimage import filters
import glob
import os
from tifffile import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .displacement import displace_slices


def _tm(image, template_im, thresh=(0, 0), sigma=1.):

    # print(f'template_im.shape = {template_im.shape}')
    # print(f'image.shape = {image.shape}')

    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
        template_im[template_im < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]
        template_im[template_im > thresh[1]] = thresh[1]
    if sigma > 0:
        if image.dtype == 'uint16':
            assert template_im.dtype == 'uint16'
            image = gaussianSmoothing(image.astype('float32'), sigma).astype('uint16')
            template_im = gaussianSmoothing(template_im.astype('float32'), sigma).astype('uint16')
        else:
            image = gaussianSmoothing(image, sigma)
            template_im = gaussianSmoothing(template_im, sigma)

    # print(f'template_im.shape = {template_im.shape}')
    # print(f'image.shape = {image.shape}')

    result = match_template(image, template_im)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    return x, y


def _wrap_tm(im_idx, im_list, im_ref, xy_range, thresh=(0, 0), sigma=1.):
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
        subpixel_displacement=False,
        threshold=(0, 0),
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
        offsets = [off - np.array(add_offset) for off in offsets]

    offsets = -np.array(offsets)

    if target_folder is not None:

        print('Displacing and saving slices ...')

        displace_slices(
            im_list, target_folder, offsets,
            subpx_displacement=subpixel_displacement,
            compression=compression,
            pad_zeros=None,
            parallel_method='multi_process',
            n_workers=n_workers
        )

    return offsets


if __name__ == '__main__':

    pass
