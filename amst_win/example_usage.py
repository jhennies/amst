from amst.amst_main import amst_align, default_amst_params

raw_folder = 'your_path\\20140801_hela-wt_xy5z8nm_as_part\\raw_8bit'
pre_alignment_folder = 'your_path\\20140801_hela-wt_xy5z8nm_as_part\\tm_pre_align'
target_folder = 'your_path\\hela_res'

# Load the default parameters
params = default_amst_params()
params['n_workers'] = 2  # The default number of CPU cores is 8; set this to the number that is available

if __name__ == '__main__':
	amst_align(raw_folder=raw_folder,
    		pre_alignment_folder=pre_alignment_folder,
    		target_folder=target_folder,
    		**params)
