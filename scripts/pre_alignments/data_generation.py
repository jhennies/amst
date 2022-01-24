import os
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from tifffile import imread
import numpy as np


def _retrieve_slice(im_fp, xy_range=np.s_[:], pre_process_func=None, pre_process_kwargs=None):

    im = imread(im_fp)[xy_range]
    if pre_process_func is not None:
        im = pre_process_func(im, **pre_process_kwargs)

    return im


def _retrieve_batch(
        im_list,
        xy_range=np.s_[:],
        pre_process_func=None,
        pre_process_kwargs=None,
        start_id=0,
        n_workers=os.cpu_count()
):

    stop_id = start_id + n_workers
    if stop_id > len(im_list):
        end_of_list = True
        stop_id = len(im_list)
    else:
        end_of_list = False

    if n_workers == 1:
        batch = [_retrieve_slice(im_fp, xy_range, pre_process_func, pre_process_kwargs) for im_fp in im_list]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_retrieve_slice, im_list[idx], xy_range, pre_process_func, pre_process_kwargs)
                for idx in range(start_id, stop_id)
            ]
            batch = [task.result() for task in tasks]

    return batch, len(batch), stop_id, end_of_list


def parallel_image_slice_generator(
        im_list,
        xy_range=np.s_[:],
        pre_process_func=None,
        pre_process_kwargs=None,
        yield_consecutive=False,
        n_workers=os.cpu_count()
):
    """
    Generator that loads and pre-processes batches of data slices in parallel.
    This is useful if the processing of data slices cannot be parallelized, e.g. for GPU jobs, while the data generation
        can be run on multiple CPUs
    :param im_list:
    :param xy_range:
    :param pre_process_func:
    :param pre_process_kwargs:
    :param yield_consecutive: Yields two consecutive slices in each iteration if set to True
    :param n_workers:
    """

    batch, batch_size, next_id, end_of_list = _retrieve_batch(
        im_list,
        xy_range=xy_range,
        pre_process_func=pre_process_func,
        pre_process_kwargs=pre_process_kwargs,
        start_id=0,
        n_workers=n_workers
    )
    if yield_consecutive:
        next_id -= 1
        batch_size -= 1

    batch_id = 0
    n = 0

    last_image_processed = False
    while not last_image_processed:

        if n == 0:
            print('Submitting new job')

            p = ThreadPool(processes=1)
            res = p.apply_async(_retrieve_batch, (
                im_list,
                xy_range,
                pre_process_func,
                pre_process_kwargs,
                next_id,
                n_workers - 1
            ))

        if n == batch_size:

            if end_of_list:
                last_image_processed = True
            else:

                n = 0
                print('Fetching results')
                batch, batch_size, next_id, end_of_list = res.get()
                if yield_consecutive:
                    next_id -= 1
                    batch_size -= 1
                print('Joining job')
                batch_id += 1

        else:

            if yield_consecutive:
                yield batch[n], batch[n + 1]
            else:
                yield batch[n]
            n += 1


if __name__ == '__main__':

    source_folder = '/media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit'

    from tifffile import imsave
    import glob
    img_list = sorted(glob.glob(os.path.join(source_folder, '*.tif')))

    # # -----------------------------------
    # # Test1:
    # # Just copy the data
    # target_folder = '/media/julian/Data/tmp/parallel_image_slice_generator_test1'
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder, exist_ok=True)
    #
    # gen = parallel_image_slice_generator(
    #     img_list,
    #     yield_consecutive=False,
    #     n_workers=os.cpu_count()
    # )
    #
    # for idx, img in enumerate(gen):
    #     print(f'idx = {idx}')
    #     imsave(os.path.join(target_folder, '{:04d}.tif'.format(idx)), data=img, compress=9)

    # # -----------------------------------
    # # Test2:
    # # Yield consecutive slices (writing two datasets which should be shifted by one slice)
    # target_folder1 = '/media/julian/Data/tmp/parallel_image_slice_generator_test2a'
    # target_folder2 = '/media/julian/Data/tmp/parallel_image_slice_generator_test2b'
    # if not os.path.exists(target_folder1):
    #     os.makedirs(target_folder1, exist_ok=True)
    # if not os.path.exists(target_folder2):
    #     os.makedirs(target_folder2, exist_ok=True)
    #
    # gen = parallel_image_slice_generator(
    #     img_list,
    #     yield_consecutive=True,
    #     n_workers=os.cpu_count()
    # )
    #
    # for idx, images in enumerate(gen):
    #     print(f'idx = {idx}')
    #     imsave(os.path.join(target_folder1, '{:04d}.tif'.format(idx)), data=images[0], compress=9)
    #     imsave(os.path.join(target_folder2, '{:04d}.tif'.format(idx)), data=images[1], compress=9)

    # -----------------------------------
    # Test3:
    # Testing pre-processing
    target_folder1 = '/media/julian/Data/tmp/parallel_image_slice_generator_test3a'
    target_folder2 = '/media/julian/Data/tmp/parallel_image_slice_generator_test3b'
    if not os.path.exists(target_folder1):
        os.makedirs(target_folder1, exist_ok=True)
    if not os.path.exists(target_folder2):
        os.makedirs(target_folder2, exist_ok=True)

    from slice_pre_processing import preprocess_slice

    gen = parallel_image_slice_generator(
        img_list,
        pre_process_func=preprocess_slice,
        pre_process_kwargs=dict(
            thresh=0,
            sigma=3.0,
            mask_range=[80, 205]
        ),
        yield_consecutive=True,
        n_workers=os.cpu_count()
    )

    for idx, images in enumerate(gen):
        print(f'idx = {idx}')
        imsave(os.path.join(target_folder1, '{:04d}.tif'.format(idx)), data=images[0], compress=9)
        imsave(os.path.join(target_folder2, '{:04d}.tif'.format(idx)), data=images[1], compress=9)
