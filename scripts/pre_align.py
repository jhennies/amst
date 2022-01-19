
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
        xcorr_threshold=0,
        xcorr_mask_range=None,
        xcorr_sigma=1.,
        template=None,
        tm_threshold=0,
        tm_sigma=0,
        tm_add_offset=None,
        n_workers=os.cpu_count(),
        rerun=False,
        verbose=False
):
    """

    :param tm_add_offset: [x, y]

    TODO:
     - Subpixel displacement for TM
     - Reduce search area for TM
    """

    from pre_alignments.xcorr import offsets_with_xcorr
    from pre_alignments.tm import offsets_with_tm

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cache_folder = os.path.join(target_folder, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    # Cross correlation alignment
    offsets_xcorr_fp = os.path.join(cache_folder, 'offsets_xcorr.pkl')

    if rerun or not os.path.exists(offsets_xcorr_fp):
        offsets_xcorr = offsets_with_xcorr(
            source_folder,
            target_folder=os.path.join(cache_folder, 'xcorr_applied') if verbose else None,  # We don't need to save the result
            xy_range=xy_range,
            z_range=z_range,
            subpixel_displacements=True,  # Used in verbose mode to displace the slices
            subtract_running_average=0,  # Has to be switched off
            threshold=xcorr_threshold,  # Defines the relevant grey value range (can be tuple to define upper and lower bound)
            mask_range=xcorr_mask_range,  # Similar to threshold, but puts everything above the upper bound to zero
            sigma=xcorr_sigma,  # Gaussian smoothing before xcorr computation
            compression=9,  # Used in verbose mode to displace the slices
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

    # Template matching alignment
    if template is None:
        print(f'Skipping template matching!')

    else:

        offsets_tm_fp = os.path.join(cache_folder, 'offsets_tm.pkl')
        if rerun or not os.path.exists(offsets_tm_fp):

            offsets_tm = offsets_with_tm(
                source_folder,
                template,
                target_folder=os.path.join(cache_folder, 'tm_applied') if verbose else None,
                xy_range=np.s_[:],
                z_range=z_range,
                subpixel_displacements=True,
                threshold=tm_threshold,
                sigma=tm_sigma,
                add_offset=tm_add_offset,
                compression=9,
                n_workers=n_workers,
                verbose=verbose
            )

            with open(offsets_tm_fp, mode='wb') as f:
                pickle.dump(offsets_tm, f)
        else:
            with open(offsets_tm_fp, mode='rb') as f:
                offsets_tm = pickle.load(f)

        raise(RuntimeError('Pre-mature end...'))

        # TODO: Implement smoothing of tm offsets (functionality of smooth displace)

        # TODO: The final offsets according to the formula OFFSETS = XCORR + smoothed(TM - XCORR)
        offsets = offsets_xcorr + _smooth_offsets(offsets_tm - offsets_xcorr)

    # TODO: Apply offsets
    _apply_offsets(offsets)

    return offsetss.cpu_count()


if __name__ == '__main__':

    source_folder = '/media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit/'
    target_folder = '/media/julian/Data/projects/misc/amst_devel/test_pre_merge/'
    template_fp = '/media/julian/Data/projects/misc/amst_devel/test_pre_merge/template.tif'

    pre_align(
        source_folder,
        target_folder,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        xcorr_threshold=0,
        xcorr_mask_range=[80, -50],
        xcorr_sigma=1.,
        template=template_fp,
        tm_threshold=[190, 255],
        tm_sigma=0,
        tm_add_offset=[4000, 200],
        n_workers=os.cpu_count(),
        rerun=False,
        verbose=True
    )
