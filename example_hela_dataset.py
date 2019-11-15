
from amst_main import amst_align
import numpy as np

raw_folder = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/raw_8bit'
pre_alignment_folder = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/tm_ali_refined'
target_folder = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/amst_pre_imod'
median_radius = 7
n_workers = 12

amst_align(
    raw_folder=raw_folder,
    pre_alignment_folder=pre_alignment_folder,
    target_folder=target_folder,
    median_radius=median_radius,
    n_workers=n_workers,
    source_range=np.s_[:700],
    with_sift=True,
    sift_params=dict(
        shift_only=True,
        subpixel_displacement=False,
        devicetype='GPU'
    ),
    elastix_params=None
)
