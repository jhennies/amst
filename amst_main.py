
import os
import numpy as np
from template_functions import parallel_median_z
from alignment_functions import alignment_function_wrapper, elastix_align_advanced, sift_align
import warnings


def amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        median_radius=7,
        n_workers=16,
        source_range=np.s_[:],
        with_sift=True,
        sift_params=None,
        elastix_params=None
):
    """
    The main function call to run Alignment to Median Smoothed Template (AMST)

    This requires the raw data and a pre-alignment to be stored as individual tif slices.

    :param raw_folder: location of folder containing the raw data in form of tif slices (*.tif)
    :param pre_alignment_folder: location of folder containing a pre-alignment as tif slices (*.tif)
    :param target_folder: where the results are saved
    :param median_radius: the radius of the z-median filter
    :param n_workers: number of cores for CPU-based computations
    :param source_range: to select a subset of the data. Use numpy.s_, e.g. for np.s_[:100] for the first 100 slices
    :return:
    """

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    if sift_params is None:
        sift_params = dict(
            shift_only=True,
            subpixel_displacement=False,
            devicetype='GPU'
        )

    if elastix_params is None:
        elastix_params = dict(
            connected_components=False,
            transform='AffineTransform',
            save_field=None,
            background_value=0,
            invert_for_align=False,
            number_of_resolutions=4,
            maximum_number_of_iterations=500,
            mode='crop_roi'
        )

    median_z_target_folder = os.path.join(
        target_folder,
        'median_z'
    )
    if not os.path.exists(median_z_target_folder):
        os.mkdir(median_z_target_folder)

    # Compute the median smoothed template
    parallel_median_z(
        source_folder=pre_alignment_folder,
        target_folder=median_z_target_folder,
        radius=median_radius,
        n_workers=n_workers,
        source_range=source_range
    )

    # SIFT to get the raw data close to the template
    if with_sift:
        sift_folder = os.path.join(
            target_folder,
            'sift'
        )
        if not os.path.exists(sift_folder):
            os.mkdir(sift_folder)
        if sift_params['devicetype'] == 'GPU':
            # Only allow one process when on GPU
            n_workers_sift = 1
        elif sift_params['devicetype'] == 'CPU':
            # For a bit of speedup on the CPU
            n_workers_sift = n_workers
        else:
            # Let's hope for the best...
            warnings.warn('Unknown device type')
            n_workers_sift = n_workers
        alignment_function_wrapper(
            func=sift_align,
            source_folder=raw_folder,
            ref_source_folder=median_z_target_folder,
            target_folder=sift_folder,
            alignment_params=sift_params,
            n_workers=n_workers_sift,
            source_range=source_range,
            ref_range=source_range
        )
        raw_folder = sift_folder

    # Affine transformations with Elastix
    alignment_function_wrapper(
        func=elastix_align_advanced,
        source_folder=raw_folder,
        ref_source_folder=median_z_target_folder,
        target_folder=target_folder,
        alignment_params=elastix_params,
        n_workers=n_workers,
        source_range=source_range,
        ref_range=source_range
    )


if __name__ == '__main__':
    pass
