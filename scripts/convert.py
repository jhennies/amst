import os

import numpy as np

from tif_stack_processing.data_modification import data_modification_pipeline
from tif_stack_processing.data_modification import to8bit, clip_values, invert, background_white_to_black

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default=None)
    parser.add_argument('--source_folder', type=str, default=None)
    parser.add_argument('--pattern', type=str, default='*.tif')
    parser.add_argument('--z_range', type=int, default=[None, None], nargs='+')
    parser.add_argument('--functions', type=str, default=None, nargs='+')
    parser.add_argument('--params', type=str, default=None, nargs='+')
    parser.add_argument('--compression', type=int, default=9)
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    target_folder = args.target_folder
    source_folder = args.source_folder
    pattern = args.pattern
    n_workers = args.n_workers
    verbose = args.verbose
    if len(args.z_range) == 2:
        zr0 = args.z_range[0]
        zr1 = args.z_range[1]
    elif len(args.z_range) == 1:
        zr0 = args.z_range[0]
        zr1 = None
    else:
        raise RuntimeError
    z_range = np.s_[zr0: zr1]
    compression = args.compression
    functions = args.functions
    params = args.params

    assert source_folder is not None
    assert target_folder is not None
    assert len(functions) == len(params)

    data_modification_pipeline(
        funcs=functions,
        params=params,
        source_folder=source_folder,
        target_folder=target_folder,
        pattern=pattern,
        compression=compression,
        z_range=z_range,
        n_workers=n_workers,
        verbose=verbose
    )
