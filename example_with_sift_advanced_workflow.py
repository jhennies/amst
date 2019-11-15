
"""
This illustrates the basic AMST workflow used to generate a high-quality alignment from a already pre-computed pre-
alignment.
The raw data is brought close to the median-smoothed template by SIFT and translations only, then Elastix takes care of 
the rest using affine transformations.

Requirements:

 - The raw data as it is
 - A pre-alignment with any alignment method

"""

# >> Imports
import sys
# sys.path.append('/path/to/where/the/amst/package/resides/')
sys.path.append('/g/schwab/hennies/src/github/amst/')

from amst import template_align
from amst import template_functions
from amst import alignment_functions

import numpy as np
import os
# << Imports


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >> Parameters -- FEEL FREE TO MODIFY HERE --

# Experiment name and data locations
# ----------------------------------

# Give the experiment a name
experiment_name = 'amst_with_sift_advanced_workflow_00'
# The raw data
source = '/path/to/empiar_download/20140801_hela-wt_xy5z8nm_as/raw_8bit/'
source = '/g/schwab/hennies/datasets/20140801_Hela-wt_xy5z8nm_AS/20140801_hela-wt_xy5z8nm_as_full_8bit/'
# The pre-aligned data
# e.g. template matching with subpixel displacement generated by Fiji plugin from:
#     https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin
template_source = '/path/to/pre_alignment/'
template_source = '/g/schwab/hennies/datasets/20140801_Hela-wt_xy5z8nm_AS/20140801_hela-wt_xy5z8nm_as_full_8bit/template_match_aligned'
# Where to save the results
results_folder = '/g/schwab/hennies/phd_project/image_analysis/alignment/amst/amst_191115_00_test_original_wf'

# General parameters
# ------------------

# Number of CPU cores
n_workers = 12

# Template parameters
# -------------------

# Radius for the median smoothing
median_radius = 7		# 7 is a quite reasonable default

# Alignment parameters
# --------------------

# For multiple regions of interest (Use if there are multiple ROIs in your image data)
connected_components = False
background_value = 0		# It is best to not invert the data before alignment such that the background value is zero
invert_for_align = False    # If the data was inverted set this parameter to True and the background_value to e.g. 255
                            #   for 8bit data, 65535 for 16bit, ...
mode = 'crop_roi'

# Elastix
transform = 'AffineTransform'		# Keep this!
number_of_resolutions = 4			# This should be fine as well
maximum_number_of_iterations = 500  # Increase if result is not satisfying, e.g. to 1500 (increases runtime!)

# Others
save_field = None

# << Parameters -- FEEL FREE TO MODIFY HERE --
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >> The function call -- DO NOT TOUCH! --

if __name__ == '__main__':

    # Where the coarse aligned data is saved
    sift_target = os.path.join(results_folder, experiment_name, 'sift')
    # Where the median smoothed data is saved
    template_target = os.path.join(results_folder, experiment_name, 'ref_ims')
    # The location for the result
    target = os.path.join(results_folder, experiment_name)

    # SIFT pre-alignment to get close to the result, the smoothed template will already be generated

    template_align.align_on_template(
        source,
        sift_target,
        source_range=np.s_[:48],
        template_source_range=np.s_[:48],
        template_target_folder=template_target,
        template_source_folder=template_source,
        make_template_function=template_align.template_functions.median_z,
        make_template_params=[(median_radius,), {}],
        alignment_function=template_align.alignment_functions.sift_align,
        alignment_params=[(), {'shift_only': True,
                               'subpixel_displacement': False}],
        n_workers=[n_workers, 1]
    )

    # Now the final elastix alignment

    template_align.align_on_template(
        sift_target,
        target,
        source_range=np.s_[:48],
        template_source_range=np.s_[:48],
        template_source_folder=template_source,
        template_target_folder=template_target,
        make_template_function=template_align.template_functions.median_z,
        make_template_params=[(median_radius,), {}],
        alignment_function=template_align.alignment_functions.elastix_align_advanced,
        alignment_params=[(), {'connected_components': connected_components,
                               'transform': transform,
                               'save_field': save_field,
                               'background_value': background_value,
                               'invert_for_align': invert_for_align,
                               'number_of_resolutions': number_of_resolutions,
                               'maximum_number_of_iterations': maximum_number_of_iterations,
                               'mode': mode}],
        n_workers=n_workers
    )

