
"""
This script only works for a windows installation.
Also only very basic parameters are exposed, if you intend to run AMST with specialized parameter settings use the
example_usage.py script
"""

import sys
sys.path.append('C:\\Users\\TEAM\\src\\new_amst\\amst')

from amst_linux.amst_main import amst_align, default_amst_params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_folder', type=str, default=None)
parser.add_argument('--pre_alignment_folder', type=str, default=None)
parser.add_argument('--target_folder', type=str, default=None)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--verbose', type=int, default=0)

args = parser.parse_args()

assert args.raw_folder is not None
assert args.pre_alignment_folder is not None
assert args.target_folder is not None

# Load the default parameters
params = default_amst_params()
params['n_workers'] = args.n_workers  # The default number of CPU cores is 8; set this to the number that is available
params['verbose'] = args.verbose

if __name__ == '__main__':
    amst_align(
        raw_folder=args.raw_folder,
        pre_alignment_folder=args.pre_alignment_folder,
        target_folder=args.target_folder,
        **params)
