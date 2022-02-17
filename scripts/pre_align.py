
import os


def pre_align(
        source_folder,
        target_folder,
        xy_range=None,
        z_range=None,
        local_threshold=(0, 0),
        local_mask_range=None,
        local_sigma=1.,
        template=None,
        tm_threshold=(0, 0),
        tm_sigma=0,
        tm_add_offset=None,
        tm_smooth_median=8,
        tm_smooth_sigma=8,
        tm_suppress_x=False,
        rerun=False,
        local_align_method='sift',
        sift_devicetype='GPU',
        n_gpus=1,
        n_workers=os.cpu_count(),
        verbose=False
):

    import numpy as np
    from pre_alignments.pre_align import pre_align_workflow

    # xy_range = [X, Y, W, H]
    xy_range = np.s_[:] if xy_range is None else np.s_[
        xy_range[1]: xy_range[1] + xy_range[3],
        xy_range[0]: xy_range[0] + xy_range[2],
    ]
    # z_range = [Z, D]
    z_range = np.s_[:] if z_range is None else np.s_[
        z_range[0]: z_range[0] + z_range[1]
    ]

    pre_align_workflow(
        source_folder,
        target_folder,
        xy_range=xy_range,
        z_range=z_range,
        local_threshold=local_threshold,
        local_mask_range=local_mask_range,
        local_sigma=local_sigma,
        template=template,
        tm_threshold=tm_threshold,
        tm_sigma=tm_sigma,
        tm_add_offset=tm_add_offset,
        tm_smooth_median=tm_smooth_median,
        tm_smooth_sigma=tm_smooth_sigma,
        tm_suppress_x=tm_suppress_x,
        rerun=rerun,
        local_align_method=local_align_method,
        sift_devicetype=sift_devicetype,
        n_gpus=n_gpus,
        n_workers=n_workers,
        verbose=verbose
    )


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Pre-alignment workflow \n'
                    'It performs a local alignment, i.e. for correspondences of adjacent slices by SIFT or cross-'
                    'correlation, as well as an alignment for large-scale correspondences using template matching.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('source_folder', type=str,
                        help='Source folder')
    parser.add_argument('target_folder', type=str,
                        help='Where to save the results')
    parser.add_argument('--xy_range', type=int, nargs=4, default=None,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Crop xy-range for computation: (x, y, width, height)')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        metavar=('Z', 'D'),
                        help='Only use certain z-range: (z, depth)')
    parser.add_argument('--local_threshold', type=float, nargs=2, default=[0, 0],
                        metavar=('lower', 'upper'),
                        help='Clip the image data below and above a certain threshold')
    parser.add_argument('--local_mask_range', type=float, nargs=2, default=None,
                        metavar=('lower', 'upper'),
                        help='Similar to threshold, except values above the upper threshold are set to zero')
    parser.add_argument('--local_sigma', type=float, default=1.,
                        help='Smooths the data before local alignment')
    parser.add_argument('--template', type=str, default=None,
                        help='Location of template tiff image. Enables template matching step if set')
    parser.add_argument('--tm_threshold', type=float, nargs=2, default=[0, 0],
                        metavar=('lower', 'upper'),
                        help='Lower and upper thresholds applied before template matching')
    parser.add_argument('--tm_sigma', type=float, default=0.,
                        help='Smooths the data before template matching alignment')
    parser.add_argument('--tm_add_offset', type=int, nargs=2, default=None,
                        metavar=('X', 'Y'),
                        help='Add an offset to the final alignment')
    parser.add_argument('--tm_smooth_median', type=int, default=8,
                        help='Median smoothing for the offsets when combining local and global alignments')
    parser.add_argument('--tm_smooth_sigma', type=float, default=8.,
                        help='Gaussian smoothing for the offsets when combining local and global alignments')
    parser.add_argument('--tm_suppress_x', action='store_true',
                        help='Suppresses x-displacements in the TM step')
    parser.add_argument('--rerun', action='store_true',
                        help='Triggers re-running of already existing results')
    parser.add_argument('--local_align_method', type=str, default='sift',
                        help='Method for local alignment: "sift" or "xcorr"')
    parser.add_argument('--sift_devicetype', type=str, default='GPU',
                        help='On which device to run the SIFT alignment: "GPU" or "CPU"')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of available GPUs')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(),
                        help='Maximum number of CPU cores to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    xy_range = args.xy_range
    z_range = args.z_range
    local_threshold = args.local_threshold
    local_mask_range = args.local_mask_range
    local_sigma = args.local_sigma
    template = args.template
    tm_threshold = args.tm_threshold
    tm_sigma = args.tm_sigma
    tm_add_offset = args.tm_add_offset
    tm_smooth_median = args.tm_smooth_median
    tm_smooth_sigma = args.tm_smooth_sigma
    tm_suppress_x = args.tm_suppress_x
    rerun = args.rerun
    local_align_method = args.local_align_method
    sift_devicetype = args.sift_devicetype
    n_gpus = args.n_gpus
    n_workers = args.n_workers
    verbose = args.verbose

    # ----------------------------------------------------

    pre_align(
        source_folder,
        target_folder,
        xy_range=xy_range,
        z_range=z_range,
        local_threshold=local_threshold,
        local_mask_range=local_mask_range,
        local_sigma=local_sigma,
        template=template,
        tm_threshold=tm_threshold,
        tm_sigma=tm_sigma,
        tm_add_offset=tm_add_offset,
        tm_smooth_median=tm_smooth_median,
        tm_smooth_sigma=tm_smooth_sigma,
        tm_suppress_x=tm_suppress_x,
        rerun=rerun,
        local_align_method=local_align_method,
        sift_devicetype=sift_devicetype,
        n_gpus=n_gpus,
        n_workers=n_workers,
        verbose=verbose
    )
