
from amst_linux.amst_main import amst_align, default_amst_params, default_elastix_params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_folder', type=str, default=None)
parser.add_argument('--pre_alignment_folder', type=str, default=None)
parser.add_argument('--target_folder', type=str, default=None)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--force_alignment', action='store_true',
                    help='Forces to align properly even if the crop from the raw data does not match the template size')
parser.add_argument('--median_radius', type=int, default=7,
                    help='Number of slices taken into account before and after the current slice for generation of the median smoothed template.')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

assert args.raw_folder is not None
assert args.pre_alignment_folder is not None
assert args.target_folder is not None

elastix_params = default_elastix_params()
elastix_params['Transform'] = "BSplineTransform"
# elastix_params['FinalGridSpacingInPhysicalUnits'] = 64
elastix_params['FinalGridSpacingInVoxels'] = 32
elastix_params['GridSpacingSchedule'] = [4.0, 4.0, 2.0, 1.0]
elastix_params['MaximumNumberOfIterations'] = 1000
elastix_params['NumberOfSpatialSamples'] = 4800
elastix_params['NumberOfResolutions'] = 4
elastix_params['Metric'] = 'NormalizedMutualInformation'
elastix_params['ImageSampler'] = 'RandomCoordinate'
elastix_params['AutomaticScalesEstimation'] = False,

# Load the default parameters
params = default_amst_params()
params['n_workers'] = args.n_workers  # The default number of CPU cores is 8; set this to the number that is available
params['verbose'] = args.verbose
params['force_alignment'] = args.force_alignment
params['median_radius'] = args.median_radius
params['elastix_params'] = elastix_params
params['coarse_alignment'] = 'xcorr'
params['n_workers_sift'] = args.n_workers
params['write_intermediates'] = True
params['normalize_images'] = True
params['gaussian_smooth'] = 1.5
# params['add_gradient'] = 1.5
params['clahe'] = [0.1, (32, 32)]


if __name__ == '__main__':
    amst_align(
        raw_folder=args.raw_folder,
        pre_alignment_folder=args.pre_alignment_folder,
        target_folder=args.target_folder,
        **params)

