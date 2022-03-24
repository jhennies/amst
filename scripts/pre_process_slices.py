
import sys
sys.path.append('/g/schwab/hennies/software/CebraEM/src/github/cebra_em/cebra_inf/inf_utils')
sys.path.append('/g/schwab/hennies/software/CebraEM/src/github/cebra_em/')

from multiprocessing import Pool
import os
from glob import glob

# from cebra_inf.inf_utils.common.bdv_utils import create_empty_dataset
# from pybdv.util import open_file, get_key
import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from tifffile import imread, imsave

from matplotlib import pyplot as plt


class QNorm:

    def __init__(self, ref_data, roi=np.s_[:], ref_mask=None, crop_to_data=False, quantile=0.1, autoclip=False, verbose=False):
        self.verbose = verbose
        self.quantile = quantile
        self.q_lower_ref, self.q_upper_ref, self.min, self.max = self._get_ref_quantiles(
            ref_data, mask=ref_mask
        )
        self._roi = roi
        self._crop_to_data = crop_to_data
        self._autoclip = autoclip
        if self._crop_to_data:
            raise NotImplementedError('Cropping to data is not implemented!')

    def _get_ref_quantiles(self, ref_data, mask=None):
        if mask is None:
            mask = ref_data != 0
        else:
            if mask.shape != ref_data.shape:
                mask = resize(mask, ref_data.shape, order=0, mode='constant')
            mask[ref_data == 0] = 0
            mask = mask > 0

        if self.verbose:
            print(f'mask.shape = {mask.shape}')
            print(f'mask.dype = {mask.dtype}')
            print(f'ref_data.shape = {ref_data.shape}')
            print(f'ref_data.dtype = {ref_data.dtype}')
            # plt.imshow(mask)
            # plt.figure()
            # plt.imshow(ref_data)
            # plt.show()

        lower, upper = np.quantile(ref_data[mask], self.quantile), np.quantile(ref_data[mask], 1 - self.quantile)

        min, max = np.quantile(ref_data[mask], self.quantile / 5) - 1, np.quantile(ref_data[mask], 1 - (self.quantile / 5)) + 1
        return lower, upper, min, max

    def _apply_quantile_normalization(self, sl, mask=None):

        if mask is None:
            mask = sl != 0
        else:
            if mask.shape != sl.shape:
                mask = resize(mask, sl.shape, order=0, mode='constant')
            mask[sl == 0] = 0
            mask = mask > 0
        if not mask.max():
            print('Warning: The mask was empty, returning the original slice')
            return sl

        if self.verbose:
            print(f'mask.shape = {mask.shape}')
            print(f'mask.dype = {mask.dtype}')
            print(f'sl.shape = {sl.shape}')
            print(f'sl.dtype = {sl.dtype}')
            # plt.imshow(mask)
            # plt.figure()
            # plt.imshow(sl)
            # plt.show()

        sl = sl.astype('float32')

        # Get quantiles of the image slice
        q_lower = np.quantile(sl[self._roi][mask[self._roi]], self.quantile)
        q_upper = np.quantile(sl[self._roi][mask[self._roi]], 1 - self.quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= self.q_upper_ref - self.q_lower_ref
        sl += self.q_lower_ref

        # Doing an auto-clip
        if self._autoclip:

            ac_min = self.q_lower_ref - (self.q_upper_ref - self.q_lower_ref) * self.quantile * 4
            ac_max = self.q_upper_ref + (self.q_upper_ref - self.q_lower_ref) * self.quantile * 4

            if verbose:
                print(f'ac_min = {ac_min}')
                print(f'ac_max = {ac_max}')
                print(f'self.q_upper_ref = {self.q_upper_ref}')
                print(f'self.q_lower_ref = {self.q_lower_ref}')

            sl -= ac_min
            sl /= ac_max - ac_min
            sl *= 255
            sl[sl < 0] = 0
            sl[sl > 255] = 255

        return sl

    def run(self, sl, mask=None):

        return self._apply_quantile_normalization(sl, mask=mask)


# class Clip:
#
#     def __init__(self, low=0, high=0, cast=None, invert=False, verbose=False):
#         self.verbose = verbose
#         self.low = low
#         self.high = high
#         self.cast = cast
#         self.invert = invert
#
#     def run(self, chunk, mask=None):
#
#         assert mask is None
#
#         if self.cast is None:
#             self.cast = chunk.dtype
#
#         if self.cast == 'uint8':
#             max_val = 255
#         elif self.cast == 'uint16':
#             max_val = 65535
#         else:
#             raise ValueError(f'invalid data type: {self.cast}')
#
#         chunk[chunk > self.high] = self.high
#         chunk = chunk - self.low
#         chunk[chunk < 0] = 0
#         chunk = (chunk.astype('float16') / (self.high - self.low) * max_val).astype(self.cast)
#         if self.invert:
#             chunk = max_val - chunk
#         return chunk


def _run_slice(im_fp, sl_idx, pp_tasks, pp_classes, target_path, dtype=None, verbose=False):

    print('\n-------------------------')
    print(f'sl_idx = {sl_idx}')

    im = imread(im_fp)

    # Run preprocessing methods
    for pp_idx, pp_task in enumerate(pp_tasks):
        func = pp_classes[pp_idx].run
        im = func(im)

    target_filepath = os.path.join(
        target_path,
        os.path.split(im_fp)[1]
    )

    if dtype is not None:
        im = im.astype(dtype)

    imsave(target_filepath, data=im)


def preprocess_slices(
        source_path,
        target_path,
        roi=np.s_[:],
        preprocess=None,
        dtype=None,
        n_workers=os.cpu_count(),
        verbose=False
):

    def _generate_tasks_from_json(params):
        tsks = []
        for tsk, kwargs in params.items():
            if tsk == 'quantile_norm':
                rsl = kwargs['ref_slice']
                del kwargs['ref_slice']
                tsks.append({
                    'class': QNorm,
                    'kwargs': kwargs,
                    'ref_slice': rsl,
                    'use_mask': True,
                    'target': 'slice'
                })
            elif tsk == 'clip':
                tsks.append({
                    'class': Clip,
                    'kwargs': kwargs,
                    'use_mask': False,
                    'target': 'slice'
                })
        return tsks

    if n_workers > 1:
        print(f'running with {n_workers} workers')

    if preprocess == 'quantile_norm':
        pp_tasks = [
            {
                'class': QNorm,
                'kwargs': {
                    'quantile': 0.1,
                    'roi': roi,
                    'crop_to_data': False,
                    'autoclip': True
                },
                'ref_slice': 0.5,
                'use_mask': False,
                'target': 'slice'  # Defines if it is run on the slice or the chunk: ['slice', 'chunk']
            }
        ]
    elif preprocess[-5:] == '.json':
        import json
        with open(preprocess, mode='r') as f:
            preprocess_params = json.load(f)
        pp_tasks = _generate_tasks_from_json(preprocess_params)
    elif preprocess is None:
        pp_tasks = []
    else:
        raise ValueError(f'Invalid value for preprocess: {preprocess}')

    # Initialize the tiff stack
    im_list = np.array(sorted(glob(os.path.join(source_path, '*.tif'))))

    # Initialize the result
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # Initialize the preprocessing methods
    print('Initializing preprocessing methods ...')
    pp_classes = []
    for pp_task in pp_tasks:
        if 'ref_slice' in pp_task.keys():
            ref_sl_id = pp_task['ref_slice']
            if ref_sl_id < 1:
                ref_sl_id = int(len(im_list) * ref_sl_id)
            ref_sl = imread(im_list[ref_sl_id])[roi]
            if verbose:
                print(f'ref_sl_id = {ref_sl_id}')
                print(f'ref_sl.shape = {ref_sl.shape}')
            pp_cl = pp_task['class'](
                ref_sl,
                ref_mask=None,  # Could be implemented here
                **pp_task['kwargs'],
                verbose=verbose
            )
        else:
            pp_cl = pp_task['class'](**pp_task['kwargs'], verbose=verbose)
        pp_classes.append(pp_cl)

    # Full resolution level (includes all preprocessing operations)
    print('Running over dataset ...')

    if n_workers == 1:

        # for sl_idx in range(0, data_shape[axis], chunks[axis]):
        #     _run_chunk(sl_idx, f_source, mask, chunks, pp_tasks, pp_classes, mask_ds_factor, verbose=verbose)

        for sl_idx, im_fp in enumerate(im_list):
            _run_slice(im_fp, sl_idx, pp_tasks, pp_classes, target_path, dtype=dtype, verbose=verbose)

    else:

        with Pool(processes=n_workers) as p:
            ts = [
                p.apply_async(_run_chunk, (
                    sl_idx, f_source, mask, chunks, pp_tasks, pp_classes, mask_ds_factor, verbose)
                )
                for sl_idx in range(0, data_shape[axis], chunks[axis])
            ]

            [t.get() for t in ts]


if __name__ == '__main__':

    # ___________________________________________________
    # Command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str,
                        help="Path of the dataset (tiff slices)")
    parser.add_argument('target_path', type=str,
                        help="Path where the result is saved. Include file name.")
    parser.add_argument('--roi', type=int, nargs=4, metavar=('x', 'y', 'w', 'h'), default=None,
                        help="Define the ROI according to Fiji's coordinate system")
    parser.add_argument('--preprocess', type=str, default=None,
                        help='Preprocessing method')
    parser.add_argument('--dtype', type=str, default=None,
                        help='Data type of the output dataset')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(),
                        help='Number of parallel slices to process')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Additional console output (for debugging purposes)")

    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    roi = args.roi
    preprocess = args.preprocess
    dtype = args.dtype
    n_workers = args.n_workers
    verbose = args.verbose
    # ___________________________________________________

    roi = np.s_[
        roi[1]: roi[1] + roi[3],
        roi[0]: roi[0] + roi[2]
    ]

    preprocess_slices(
        source_path,
        target_path,
        roi=roi,
        preprocess=preprocess,
        dtype=dtype,
        n_workers=n_workers,
        verbose=verbose
    )
