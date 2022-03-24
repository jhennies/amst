import numpy as np
from skimage.exposure import equalize_adapthist


def vahe(
        vol, clip_limit=0.9, kernel_size=(32, 32, 32)
):
    """
    Volume adaptive histogram equalization (volumetric AHE filter)
    """

    # # Normalize image to a range from 0 to 1
    # vol = np.clip(vol, np.percentile(vol, 5), np.percentile(vol, 95))
    # vol = (vol - vol.min()) / (vol.max() - vol.min())

    # Perform histogram equalization
    vol = equalize_adapthist(vol, kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)
    return vol
