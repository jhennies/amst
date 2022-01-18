
import os
import numpy as np
from multiprocessing import Pool
from tifffile import imread, imsave
from scipy.signal import medfilt2d
from glob import glob
from matplotlib import pyplot as plt


def to8bit(im):
    assert im.dtype == 'uint16', 'Only implemented for 16bit data!'
    im = im.astype('float32')
    im /= 2 ** 16
    im *= 2 ** 8
    im = im.astype('uint8')
    return im


def clip_values(im, min_value, max_value):
    assert im.dtype == 'uint8', 'Only implemented for 8bit data!'
    im = im.astype('float32')
    im[im < min_value] = min_value
    im[im > max_value] = max_value
    im -= min_value
    im = (im/im.max()) * 255
    return im.astype('uint8')


def invert(im, max_value=None):
    if max_value is not None:
        im = max_value - im
    else:
        im = im.max() - im

    return im


def background_white_to_black(im, kernel_size):

    assert im.dtype == 'uint8'
    mask = (im == 255).astype('uint8') * 255
    mask = medfilt2d(mask, kernel_size)
    return im - mask


def _evaluate_params(params):
    if params == 'None':
        return []
    params = params.split(',')
    return_params = []
    for idx, param in enumerate(params):
        try:
            return_params.append(float(param))
        except ValueError:
            return_params.append(param)

    return return_params


def _wrap_slice(funcs, params, source_filepath, target_folder, compression, verbose=False):

    target_filepath = os.path.join(
        target_folder,
        os.path.split(source_filepath)[1]
    )
    print(target_filepath)

    im = imread(source_filepath)

    if funcs is not None:
        if verbose:
            print(f'params = {params}')
        for fid, func in enumerate(funcs):
            p = params[fid]
            if verbose:
                print(f'p = {p}')
            if type(p) == str:
                p = _evaluate_params(p)
            if verbose:
                print(f'p = {p}')
            if type(func) == str:
                func = eval(func)
            im = func(im, *p)

    imsave(target_filepath, data=im, compress=compression)


def data_modification_pipeline(
        funcs,
        params,
        source_folder,
        target_folder,
        pattern='slice*.tif',
        compression=9,
        z_range=np.s_[:],
        n_workers=os.cpu_count(),
        verbose=False

):
    assert source_folder != target_folder, 'The results folder must differ from the input folder!'
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    im_list = sorted(glob(os.path.join(source_folder, pattern)))[z_range]
    # if verbose:
    #     print(im_list)

    if n_workers == 1:

        [_wrap_slice(funcs, params, source, target_folder, compression, verbose) for source in im_list]

    else:

        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(
                    _wrap_slice, (funcs, params, source, target_folder, compression, verbose)
                )
                for source in im_list
            ]
            [task.get() for task in tasks]
