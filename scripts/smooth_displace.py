
import sys

from pre_alignments.displacement import smooth_displace
import numpy as np
import os
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_folder', type=str, default=None)
parser.add_argument('--source_folder', type=str, default=None)
parser.add_argument('--displacement_file', type=str, default=None)
parser.add_argument('--median', type=int, default=0)
parser.add_argument('--gauss', type=float, default=0.0)
parser.add_argument('--n_workers', type=int, default=os.cpu_count())
parser.add_argument('--pattern', type=str, default='*.tif')
parser.add_argument('--suppress_x', type=int, default=0)
parser.add_argument('--source_range', type=int, default=[None, None], nargs='+')

args = parser.parse_args()
target_folder = args.target_folder
source_folder = args.source_folder
displacement_file = args.displacement_file
median = args.median
gauss = args.gauss
n_workers = args.n_workers
pattern = args.pattern
suppress_x = args.suppress_x
if len(args.source_range) == 2:
    sr0 = args.source_range[0]
    sr1 = args.source_range[1]
elif len(args.source_range) == 1:
    sr0 = args.source_range[0]
    sr1 = None
else:
    raise RuntimeError
source_range = np.s_[sr0: sr1]

assert source_folder is not None
assert target_folder is not None
assert displacement_file is not None

if not os.path.exists(target_folder):
    os.mkdir(target_folder)

if __name__ == '__main__':

    smooth_displace(
        source_folder=source_folder,
        target_folder=target_folder,
        displacements_file=displacement_file,
        median_radius=median,
        gaussian_sigma=gauss,
        n_workers=n_workers,
        source_range=source_range,
        parallel_method='multi_process',
        suppress_x=bool(suppress_x),
        pattern=pattern,
        compression=9,
        verbose=2
    )

    plt.show()
