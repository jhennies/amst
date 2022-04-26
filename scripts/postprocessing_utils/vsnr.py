
from pyvsnr import VSNR
import numpy as np
from scripts.pre_alignments.data_generation import _crop_zero_padding


def _norm(im, quantiles):
    im = im.astype('float32')
    # upper = np.quantile(im[im > 0], quantiles[1])
    # lower = np.quantile(im[im > 0], quantiles[0])
    # # lower = im[im > 0].min()
    # # print(f'lower = {lower}')
    # # upper = im.max()
    # # print(f'upper = {upper}')
    # im -= lower
    # im /= (upper - lower)
    # im[im < 0] = 0
    # im[im > 1] = 1
    # im *= 65535
    im /= im.max()
    return im


def vsnr(img, alpha=1e-2, filter='gabor', sigma=(1, 30), theta=0, maxit=100, is_gpu=True, verbose=False):

    if verbose:
        print(f'img.dtype = {img.dtype}')
    return_vol = False
    if img.ndim == 3:
        assert img.shape[0] == 1
        img = img[0, :]
        return_vol = True
    assert img.ndim == 2
    return_shape = img.shape

    bounds = _crop_zero_padding(img)
    img = img[bounds]

    # img = _norm(img, (0.01, 0.999999))

    # Inside the VSNR package there sometimes is a problem with the fft if the image shape is odd
    # -> Ensuring that the shape is even in all dimensions and cropping it back at the end of this function.
    crop_shape = img.shape
    img_ = np.zeros((np.ceil(np.array(img.shape) / 2) * 2).astype(int), dtype=img.dtype)
    img_[:img.shape[0], :img.shape[1]] = img
    img = img_

    img = img.astype('float32')
    imax = img.max()
    if verbose:
        print(f'imax = {imax}')
    img /= imax

    # vsnr object creation
    vsnr = VSNR(img.shape)

    if type(alpha) != list:
        # add filter (at least one !)
        # vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
        vsnr.add_filter(alpha=alpha, name=filter, sigma=sigma, theta=theta)
        #
        # vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
        # vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40), theta=20)
    else:
        for idx, a in enumerate(alpha):
            vsnr.add_filter(alpha=a, name=filter, sigma=sigma[idx], theta=theta[idx])

    # vsnr initialization
    vsnr.initialize(is_gpu=is_gpu)

    # image processing
    img = vsnr.eval(img, maxit=maxit)  # , cvg_threshold=1e-4)
    img = np.clip(img, 0, 1) * imax

    img_ = np.zeros(return_shape, dtype=img.dtype)
    img_[bounds] = img[:crop_shape[0], :crop_shape[1]]
    img = img_
    # img = img[:return_shape[0], :return_shape[1]]

    if return_vol:
        return img[None, :]
    else:
        return img
