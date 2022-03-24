
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
 3. Use smooth_displace.py to apply the TM displacements with the required smoothing
"""

import os
import numpy as np
import pickle
import glob


def pre_align_workflow(
        source_folder,
        target_folder,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        local_threshold=0,
        local_mask_range=None,
        local_sigma=1.,
        local_norm_quantiles=(0.1, 0.9),
        template=None,
        tm_threshold=0,
        tm_sigma=0,
        tm_add_offset=None,
        tm_smooth_median=8,
        tm_smooth_sigma=8,
        tm_suppress_x=False,
        rerun=False,
        local_align_method='sift',
        sift_devicetype='GPU',
        auto_pad=False,
        n_gpus=1,
        n_workers=os.cpu_count(),
        verbose=False
):
    """

    :param template: A template for the "template matching" (TM) alignment.
        If template=None no template matching is performed
    :param tm_add_offset: [x, y]
    :param local_align_method: "sift" or "xcorr"
    :param sift_devicetype: "GPU" or "CPU"
    :param n_gpus: number of available GPUs (This is probably not working for multiple GPUs yet)
    :param n_workers: number of CPUs

    TODO:
     - Subpixel displacement for TM
     - Reduce search area for TM
     - Auto detection for add_offset (from reference slice)
     - Mask for SIFT (run only on non-zero region) -> Does this make it quicker?
     - Make run reproducible by dumping all relevant run information into the target folder:
        > Function and parameters
          -> This can already be done manually by wrapping the python scripts/pre_align.py call into a bash script
        > Repo information: branch, commit, ...
     - Implement batch processing for saving of intermediate results
    """

    from .tm import offsets_with_tm
    from .displacement import smooth_offsets, displace_slices

    def _local_alignment(method):
        if method == 'xcorr':
            from .xcorr import offsets_with_xcorr
            return offsets_with_xcorr(
                source_folder,
                target_folder=os.path.join(cache_folder, 'xcorr_applied') if verbose else None,
                xy_range=xy_range,
                z_range=z_range,
                subpixel_displacement=True,  # Used in verbose mode to displace the slices
                subtract_running_average=0,  # Has to be switched off
                threshold=local_threshold,  # Defines the relevant grey value range (can be tuple to define upper and lower bound)
                mask_range=local_mask_range,  # Similar to threshold, but puts everything above the upper bound to zero
                sigma=local_sigma,  # Gaussian smoothing before xcorr computation
                compression=9,  # Used in verbose mode to displace the slices
                return_sequential=True,
                n_workers=n_workers,
                verbose=verbose
            )
        elif method == 'sift':
            from .sift import offsets_with_sift
            return offsets_with_sift(
                source_folder,
                target_folder=os.path.join(cache_folder, 'sift_applied') if verbose else None,
                xy_range=xy_range,
                z_range=z_range,
                subtract_running_average=0,
                subpixel_displacement=True,
                threshold=local_threshold,
                mask_range=local_mask_range,
                sigma=local_sigma,
                norm_quantiles=local_norm_quantiles,
                compression=9,
                return_sequential=True,
                devicetype=sift_devicetype,
                return_bounds=auto_pad,
                n_workers=n_workers,
                n_gpus=n_gpus,
                verbose=verbose
            )
        else:
            raise ValueError(f'Invalid method: {method}')

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cache_folder = os.path.join(target_folder, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    # Alignment step for local correspondences (can be SIFT or XCORR)
    offsets_local_fp = os.path.join(cache_folder, f'offsets_{local_align_method}.pkl')
    if rerun or not os.path.exists(offsets_local_fp):
        offsets_local = _local_alignment(local_align_method)
        with open(offsets_local_fp, mode='wb') as f:
            pickle.dump(offsets_local, f)
    else:
        with open(offsets_local_fp, mode='rb') as f:
            offsets_local = pickle.load(f)

    bounds = None
    if auto_pad:
        bounds = offsets_local[1]
        offsets_local = offsets_local[0]

    if verbose:
        print(f'offsets_local = {offsets_local}')

    # Template matching alignment
    if template is None:
        print(f'Skipping template matching!')
        offsets = offsets_local

    else:

        offsets_tm_fp = os.path.join(cache_folder, 'offsets_tm.pkl')
        if rerun or not os.path.exists(offsets_tm_fp):

            offsets_tm = offsets_with_tm(
                source_folder,
                template,
                target_folder=os.path.join(cache_folder, 'tm_applied') if verbose else None,
                xy_range=np.s_[:],
                z_range=z_range,
                subpixel_displacement=True,
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

        # The final offsets according to the formula OFFSETS = LOCAL + smoothed(TM - LOCAL)
        offsets = offsets_local + smooth_offsets(
            offsets_tm - offsets_local,
            median_radius=tm_smooth_median,
            gaussian_sigma=tm_smooth_sigma,
            suppress_x=tm_suppress_x
        )

    # Apply final offsets
    im_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))[z_range]
    result_folder = os.path.join(target_folder, 'pre_align')
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    displace_slices(
        im_list, result_folder, offsets,
        subpx_displacement=True,
        compression=9,
        pad_zeros=None,
        bounds=bounds,
        parallel_method='multi_process',
        n_workers=n_workers
    )

    return offsets


if __name__ == '__main__':

    # # --------------------------------------------------
    # # Test 1: SIFT and TM
    #
    # source_folder = '/media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit/'
    # target_folder = '/media/julian/Data/projects/misc/amst_devel/test_pre_align_sift2/'
    # template_fp = '/media/julian/Data/projects/misc/amst_devel/test_pre_align/template.tif'
    #
    # pre_align(
    #     source_folder,
    #     target_folder,
    #     xy_range=np.s_[:],
    #     z_range=np.s_[:],
    #     local_threshold=0,
    #     local_mask_range=[80, -50],
    #     local_sigma=1.6,
    #     template=template_fp,
    #     tm_threshold=[190, 255],
    #     tm_sigma=0,
    #     tm_add_offset=[4000, 200],
    #     tm_smooth_median=8,
    #     tm_smooth_sigma=8,
    #     n_workers=os.cpu_count(),
    #     local_align_method='sift',
    #     sift_devicetype='GPU',
    #     rerun=False,
    #     verbose=True
    # )

    # --------------------------------------------------
    # Test 2: SIFT only

    source_folder = '/media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit/'
    target_folder = '/media/julian/Data/projects/misc/amst_devel/test_pre_align_sift_only/'

    pre_align(
        source_folder,
        target_folder,
        xy_range=np.s_[:],
        z_range=np.s_[:],
        local_threshold=0,
        local_mask_range=[80, -50],
        local_sigma=1.6,
        template=None,
        n_workers=os.cpu_count(),
        local_align_method='sift',
        sift_devicetype='GPU',
        rerun=False,
        verbose=False
    )

