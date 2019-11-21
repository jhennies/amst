
from amst_main import amst_align, default_amst_params

raw_folder = '/path/to/raw/data/'
pre_alignment_folder = '/path/to/pre/alignment'
target_folder = '/where/to/save/the/results'

# Load the default parameters
params = default_amst_params()
params['n_workers'] = 12  # The default number of CPU cores is 8; set this to the number that is available

amst_align(
    raw_folder=raw_folder,
    pre_alignment_folder=pre_alignment_folder,
    target_folder=target_folder,
    **params
)
