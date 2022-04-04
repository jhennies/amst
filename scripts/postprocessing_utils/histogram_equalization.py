import numpy as np
from skimage.exposure import equalize_adapthist


def vahe(
        vol, clip_limit=0.9, kernel_size=(32, 32, 32), verbose=False
):
    """
    Volume adaptive histogram equalization (volumetric AHE filter)
    """

    # # Normalize image to a range from 0 to 1
    # vol = np.clip(vol, np.percentile(vol, 5), np.percentile(vol, 95))
    # vol = (vol - vol.min()) / (vol.max() - vol.min())

    if len(kernel_size) == 3:

        # Perform histogram equalization
        vol = equalize_adapthist(vol, kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)

    elif len(kernel_size) == 2:
        if vol.ndim == 3:
            assert vol.shape[0] == 1
            vol = equalize_adapthist(vol[0, :], kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)[None, :]
        else:
            vol = equalize_adapthist(vol, kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)
    return vol
