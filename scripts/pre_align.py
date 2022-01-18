
"""
Pipeline for pre-alignment implementing the following steps:

 1. Alignment for adjacent slice correspondence by either SIFT or cross-correlation (XCORR)
 2. Alignment for long-distance correspondence by template matching (TM)

The same workflow can be performed using FIJI by following these steps:
 1. Align the dataset with "Linear stack alignment with SIFT" plugin
 2. Run this plugin: https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin
    Input is the result of step 1
    Save only the displacement file
    Here, a cropped region of the data can be used to speed up the process and reduce memory usage
 3. Use smooth_displace.py to apply the displacements
"""

import os
import numpy as np
import pickle


def pre_align(
        source_folder,
        target_folder,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        threshold=0,
        sigma=1.,
        n_workers=os.cpu_count(),
        verbose=False
):

    from .pre_alignments.xcorr import offsets_with_xcorr
    from .pre_alignments.tm import offsets_with_tm

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cache_folder = os.path.join(target_folder, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    offsets_xcorr_fp = os.path.join(cache_folder, 'offsets_xcorr.pkl')

    if not os.path.exists(offsets_xcorr_fp):
        offsets_xcorr = offsets_with_xcorr(
            source_folder,
            target_folder=None,  # We don't need to save the result
            xy_range=xy_range,
            z_range=z_range,
            subtract_running_average=0,  # Has to be switched off
            threshold=threshold,  # Defines the relevant grey value range (can be tuple to define upper and lower bound)
            sigma=sigma,  # Gaussian smoothing before xcorr computation
            return_sequential=True,
            n_workers=n_workers,
            verbose=verbose
        )
        with open(offsets_xcorr_fp, mode='wb') as f:
            pickle.dump(offsets_xcorr, f)
    else:
        with open(offsets_xcorr_fp, mode='rb') as f:
            offsets_xcorr = pickle.load(f)

    if verbose:
        print(f'offsets_xcorr = {offsets_xcorr}')

    raise(RuntimeError('Pre-mature end...'))

    # TODO: this function
    offsets_tm = offsets_with_tm(
        source_folder,
        target_folder=target_folder,
        xy_range=
    )

    # TODO: Implement smoothing of tm offsets (functionality of smooth displace)

    # TODO: The final offsets according to the formula OFFSETS = XCORR + smoothed(TM - XCORR)
    offsets = offsets_xcorr + _smooth_offsets(offsets_tm - offsets_xcorr)

    # TODO: Apply offsets
    _apply_offsets(offsets)

    return offsets


if __name__ == '__main__':

    source_folder = '/media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1'
    target_folder = '/media/julian/Data/projects/misc/amst_devel/test_pre_merge/'

    pre_align(
        source_folder,
        target_folder,
        xy_range=np.s_[:],
        z_range=np.s_[:100],
        threshold=0,
        sigma=1.,
        n_workers=os.cpu_count(),
        verbose=True
    )
