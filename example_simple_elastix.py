
from amst_main import amst_align
import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

raw_folder = '/data/datasets/empiar_upload/20140801_hela-wt_xy5z8nm_as_part/raw_8bit'
pre_alignment_folder = '/data/datasets/empiar_upload/20140801_hela-wt_xy5z8nm_as_part/tm_pre_align'
target_folder = '/data/datasets/empiar_upload/tmp/amst_test2/'
median_radius = 7
n_workers = 12

amst_align(
    raw_folder=raw_folder,
    pre_alignment_folder=pre_alignment_folder,
    target_folder=target_folder,
    median_radius=median_radius,
    n_workers=n_workers,
    source_range=np.s_[:50],
    with_sift=True,
    sift_params=dict(
        shift_only=True,
        subpixel_displacement=False,
        devicetype='CPU'
    ),
    elastix_params=None
)
