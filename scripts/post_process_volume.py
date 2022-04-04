
import numpy as np
import os
from glob import glob
from tifffile import imread, imsave
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def _post_process_batch(
        batch_id, im_list, batch_halo, batch_size, im_shape, im_dtype,
        postprocess, postprocess_args, roi_xy, target_path, dtype,
        verbose=False
):
    print('-----------------------------------')
    print(f'batch_id = {batch_id}')

    # Checking if results exist
    existing = np.array([
        os.path.exists(
            os.path.join(
                target_path,
                os.path.split(im_list[idx])[1]
            )
        )
        for idx in range(batch_id, batch_id + batch_size)
    ])
    if existing.min():
        print('Batch results exist!')
        return

    # Select the relevant image slices (padd with None in case of halo out of bounds)
    start = batch_id - batch_halo
    stop = batch_id + batch_size + batch_halo
    batch_list = []
    if start < 0:
        batch_list.extend([None] * -start)
        start = 0
    if stop > len(im_list):
        batch_list.extend(im_list[start:])
        batch_list.extend([None] * (stop - len(im_list)))
    else:
        batch_list.extend(im_list[start: stop])

    if verbose:
        print(f'batch_list = {batch_list}')
        print(f'len(batch_list) = {len(batch_list)}')

    # Populate the batch with data
    batch = []
    for im_fp in batch_list:

        if im_fp is None:
            batch.append(np.zeros(im_shape, dtype=im_dtype))
        else:
            if verbose:
                print(f'reading: {im_fp}')
            batch.append(imread(im_fp)[roi_xy])

    batch = np.array(batch)
    if verbose:
        print(f'batch.shape = {batch.shape}')

    # Apply the post-processing functions
    if type(postprocess) != list:
        postprocess = [postprocess]
        assert type(postprocess_args) != list
        postprocess_args = [postprocess_args]
    for pp_idx, pp in enumerate(postprocess):
        if verbose:
            print(f'applying: {pp}')
            print(f'with arguments: {postprocess_args[pp_idx]}')
        batch = pp(batch, **postprocess_args[pp_idx], verbose=verbose)

    # Save the results (crop to halo!)
    if batch_halo > 0:
        batch = batch[batch_halo: -batch_halo, :]
        batch_list = batch_list[batch_halo: -batch_halo]
    for slid, sl in enumerate(batch):
        if batch_list[slid] is not None:
            out_fp = os.path.join(
                target_path,
                os.path.split(batch_list[slid])[1]
            )
            imsave(out_fp, data=sl.astype(dtype))


def post_process_volume(
        source_path,
        target_path,
        roi_xy=np.s_[:],
        roi_z=np.s_[:],
        postprocess=None,
        postprocess_args=None,
        batch_size=1,
        batch_halo=0,
        dtype=None,
        n_workers=os.cpu_count(),
        parallel_method='process',  # in ['process', 'thread']
        verbose=False
):

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    im_list = sorted(glob(os.path.join(source_path, '*.tif')))[roi_z]

    if verbose:
        print(f'im_list = {im_list}')

    # Determine the image shape and dtype
    im0 = imread(im_list[0])[roi_xy]
    im_shape = im0.shape
    im_dtype = im0.dtype
    del im0

    dtype = im_dtype if dtype is None else dtype

    if n_workers == 1:
        print('Running with 1 worker ...')
        for batch_id in range(0, len(im_list), batch_size):

            _post_process_batch(
                batch_id, im_list, batch_halo, batch_size, im_shape, im_dtype,
                postprocess, postprocess_args, roi_xy, target_path,
                dtype, verbose=verbose
            )

    else:

        if parallel_method == 'thread':
            print(f'Running with {n_workers} threads ...')
            with ThreadPoolExecutor(max_workers=n_workers) as tpe:
                tasks = [
                    tpe.submit(
                        _post_process_batch, batch_id, im_list, batch_halo, batch_size, im_shape, im_dtype,
                        postprocess, postprocess_args, roi_xy, target_path,
                        dtype, verbose
                    )
                    for batch_id in range(0, len(im_list), batch_size)
                ]
                [task.result() for task in tasks]
        elif parallel_method == 'process':
            print(f'Running with {n_workers} processes ...')
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        _post_process_batch, (
                            batch_id, im_list, batch_halo, batch_size, im_shape, im_dtype,
                            postprocess, postprocess_args, roi_xy, target_path,
                            dtype, verbose
                        )
                    )
                    for batch_id in range(0, len(im_list), batch_size)
                ]
                [task.get() for task in tasks]
        else:
            raise ValueError(f'parallel_method="{parallel_method}" not in ["thread", "process"]')


if __name__ == '__main__':

    # ___________________________________________________
    # Command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str,
                        help="Path of the dataset (tiff slices)")
    parser.add_argument('target_path', type=str,
                        help="Path where the result is saved. Include file name.")
    parser.add_argument('--roi_xy', type=int, nargs=4, metavar=('x', 'y', 'w', 'h'), default=None,
                        help="Define the ROI according to Fiji's coordinate system")
    parser.add_argument('--roi_z', type=int, nargs=2, metavar=('z', 'd'), default=None,
                        help="Range along z-stack used for computation")
    parser.add_argument('--postprocess', nargs='+', type=str, default=None,
                        help='Postprocessing method(s)')
    parser.add_argument('--postprocess_args', type=str, default=None,
                        help='json file containing arguments for each function specified in postprocess')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of slices processes as a volume')
    parser.add_argument('--batch_halo', type=int, default=0,
                        help='Overlap to compensate for boundary effects of 3D processing methods')
    parser.add_argument('--dtype', type=str, default=None,
                        help='Data type of the output dataset')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(),
                        help='Number of parallel slices to process')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Additional console output (for debugging purposes)")

    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    roi_xy = args.roi_xy
    roi_z = args.roi_z
    postprocess = args.postprocess
    postprocess_args = args.postprocess_args
    batch_size = args.batch_size
    batch_halo = args.batch_halo
    dtype = args.dtype
    n_workers = args.n_workers
    verbose = args.verbose
    # ___________________________________________________

    if roi_xy is not None:
        roi_xy = np.s_[
            roi_xy[1]: roi_xy[1] + roi_xy[3],
            roi_xy[0]: roi_xy[0] + roi_xy[2]
        ]
    else:
        roi_xy = np.s_[:]
    if roi_z is not None:
        roi_z = np.s_[
            roi_z[0]: roi_z[0] + roi_z[1]
        ]
    else:
        roi_z = np.s_[:]

    def pp_funcs(pp):
        import json
        if pp == 'vahe':
            from postprocessing_utils.histogram_equalization import vahe
            return vahe
        if pp == 'vsnr':
            from postprocessing_utils.vsnr import vsnr
            return vsnr
        else:
            raise ValueError(f'Invalid post-processing function: {pp}')
    if type(postprocess) == str or (type(postprocess) == list and len(postprocess) == 1):
        if type(postprocess) == list:
            postprocess = postprocess[0]
        postprocess = pp_funcs(postprocess)
    elif type(postprocess) == list:
        postprocess = [pp_funcs(pp) for pp in postprocess]
    elif postprocess is None:
        pass
    else:
        raise ValueError(f'Invalid post-processing argument!')

    if postprocess_args is not None:
        import json
        with open(postprocess_args, mode='r') as f:
            postprocess_args = json.load(f)

    post_process_volume(
        source_path,
        target_path,
        roi_xy=roi_xy,
        roi_z=roi_z,
        postprocess=postprocess,
        postprocess_args=postprocess_args,
        batch_size=batch_size,
        batch_halo=batch_halo,
        dtype=dtype,
        n_workers=n_workers,
        verbose=verbose
    )
