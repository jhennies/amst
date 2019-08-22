
import os
from . import template_functions
from . import alignment_functions
import numpy as np
import logging

import warnings


def align_on_template(
        source_folder, target_folder,
        template_source_folder=None,
        template_target_folder=None,
        make_template_function=template_functions.median_z,
        make_template_params=None,
        alignment_function=alignment_functions.sift_align,
        alignment_params=None,
        source_range=np.s_[:],
        template_source_range=np.s_[:],
        n_workers=1
):

    if make_template_params is None:
        make_template_params = template_functions.defaults(make_template_function)
    if alignment_params is None:
        alignment_params = alignment_functions.defaults(alignment_function)
    if type(n_workers) is int:
        n_workers = [n_workers, n_workers]
    if template_source_folder is None:
        template_source_folder = source_folder
    if template_target_folder is None:
        template_target_folder = os.path.join(target_folder, 'template')

    if not os.path.exists(target_folder):
        try:
            os.mkdir(target_folder)
        except FileNotFoundError:
            warnings.warn('Making target directory failed, trying with parent folders...')
            os.makedirs(target_folder, exist_ok=True)

    # Create the folder for the template dataset
    if not os.path.exists(template_target_folder):
        os.mkdir(template_target_folder)

    logging.info('Making reference dataset')
    # Make the reference dataset
    template_functions.template_function_wrapper(
        make_template_function, template_source_folder, template_target_folder,
        *make_template_params[0],
        n_workers=n_workers[0],
        source_range=template_source_range,
        **make_template_params[1]
    )

    logging.info('Aligning datasets')
    # Align both datasets
    alignment_functions.alignment_function_wrapper(
        alignment_function, source_folder, template_target_folder, target_folder,
        *alignment_params[0],
        n_workers=n_workers[1],
        source_range=source_range,
        **alignment_params[1]
    )