
from amst_main import amst_align
import numpy as np

raw_folder = '/g/schwab/hennies/datasets/empiar_upload/20140801_hela-wt_xy5z8nm_as/raw_8bit'
pre_alignment_folder = '/g/schwab/hennies/phd_project/image_analysis/alignment/template_align/amst191113_00_hela_record_tm_displacements/tm/slices/'
target_folder = '/g/schwab/hennies/phd_project/image_analysis/alignment/template_align/amst191113_00_hela_record_tm_displacements/amst/'
displacements = '/g/schwab/hennies/phd_project/image_analysis/alignment/template_align/amst191113_00_hela_record_tm_displacements/tm/displacements.csv'
median_radius = 7
n_workers = 16


amst_align(
    raw_folder=raw_folder,
    pre_alignment_folder=pre_alignment_folder,
    target_folder=target_folder,
    median_radius=median_radius,
    n_workers=n_workers,
    source_range=np.s_[:200],
    displacements_file=displacements,
    with_sift=False,
    sift_params=None,  # Not needed
    elastix_params=None  # Use defaults
)
