
import numpy as np
import os
from h5py import File
from tifffile import imread, imsave
from multiprocessing import Pool
from glob import glob


def normalize_slices_with_quantiles(volume, quantile=0.05):

    dtype = volume.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    volume = volume.astype('float64')

    # Get quantiles of full volume
    # Could potentially also be a reference slice, multiple reference slices, ...
    q_lower_ref = np.quantile(volume, quantile)
    q_upper_ref = np.quantile(volume, 1 - quantile)

    # Process slices
    # This can be parallelized
    for slid, sl in enumerate(volume):

        # Get quantiles of the image slice
        q_lower = np.quantile(sl, quantile)
        q_upper = np.quantile(sl, 1 - quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        volume[slid] = sl

    # Clip everything that went out of range
    # FIXME this assumes dtype==uint8
    volume[volume < 0] = 0
    volume[volume > 255] = 255

    # Convert back to the original dtype
    return volume.astype(dtype)


def normalize_tif_with_quantiles(
        filepath, target_folder, q_lower_ref, q_upper_ref, quantile,
        mask_filepath=None, min_no_of_items=64):

    sl = imread(filepath)
    if mask_filepath is not None:
        mask = imread(mask_filepath).astype(bool)
    else:
        mask = None
    target_filepath = os.path.join(
        target_folder,
        os.path.split(filepath)[1]
    )

    if not os.path.isfile(target_filepath):

        print('processing: {}'.format(target_filepath))

        assert sl.dtype == 'uint8'
        sl = sl.astype('float64')

        # Get quantiles of the image slice
        if mask is None:
            q_lower = np.quantile(sl, quantile)
            q_upper = np.quantile(sl, 1 - quantile)
        else:
            sl_masked = sl[mask]
            if len(sl_masked > min_no_of_items):
                q_lower = np.quantile(sl_masked, quantile)
                q_upper = np.quantile(sl_masked, 1 - quantile)
                print('q_upper = {}, q_lower = {}'.format(q_upper, q_lower))
            else:
                imsave(target_filepath, data=sl.astype('uint8'))
                print('Not enough items')
                return

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        # Clip everything that went out of range
        sl[sl < 0] = 0
        sl[sl > 255] = 255

        imsave(target_filepath, data=sl.astype('uint8'))

    else:

        print('exists: {}'.format(target_filepath))


def normalize_h5_with_quantiles(filepath, target_folder, q_lower_ref, q_upper_ref, quantile):

    with File(filepath, mode='r') as f:
        sl = f['data'][:]

    target_filepath = os.path.join(
        target_folder,
        os.path.split(filepath)[1]
    )

    if not os.path.isfile(target_filepath):

        print('processing: {}'.format(target_filepath))

        assert sl.dtype == 'uint8'
        sl = sl.astype('float64')

        # Get quantiles of the image slice
        q_lower = np.quantile(sl, quantile)
        q_upper = np.quantile(sl, 1 - quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        # Clip everything that went out of range
        sl[sl < 0] = 0
        sl[sl > 255] = 255

        with File(target_filepath, mode='w') as f:
            f.create_dataset('data', data=sl.astype('uint8'), compression='gzip')

    else:

        print('exists: {}'.format(target_filepath))


class QuantileNormalizer:

    def __init__(
            self,
            ref_data,
            quantile=0.05,
            mask=None,
            min_no_of_items=64
    ):

        self.quantile = quantile
        if mask is None:
            self.q_lower = np.quantile(ref_data, quantile)
            self.q_upper = np.quantile(ref_data, 1 - quantile)
        else:
            self.q_lower = np.quantile(ref_data[mask], quantile)
            self.q_upper = np.quantile(ref_data[mask], 1 - quantile)
        self.min_no_of_items = min_no_of_items
        print('q_lower_ref = {}, q_upper_ref = {}'.format(self.q_lower, self.q_upper))

    def run_on_tif_stack(self, folder, target_folder, n_workers=1, mask_folder=None):

        im_list = np.sort(glob(os.path.join(folder, '*.tif')))
        if mask_folder is not None:
            mask_list = np.sort(glob(os.path.join(mask_folder, '*.tif')))

            if n_workers == 1:
                [normalize_tif_with_quantiles(
                    fp, target_folder, self.q_lower, self.q_upper, self.quantile,
                    mask_filepath=mask_list[fp_idx],
                    min_no_of_items=self.min_no_of_items
                )
                 for fp_idx, fp in enumerate(im_list)]

            else:
                print('{} workers'.format(n_workers))
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            normalize_tif_with_quantiles, (
                                fp, target_folder, self.q_lower, self.q_upper, self.quantile, mask_list[fp_idx],
                                self.min_no_of_items
                            )
                        )
                        for fp_idx, fp in enumerate(im_list)
                    ]
                [task.get() for task in tasks]

        else:

            if n_workers == 1:
                [normalize_tif_with_quantiles(
                    fp, target_folder, self.q_lower, self.q_upper, self.quantile
                )
                    for fp_idx, fp in enumerate(im_list)]

            else:
                print('{} workers'.format(n_workers))
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            normalize_tif_with_quantiles, (
                                fp, target_folder, self.q_lower, self.q_upper, self.quantile
                            )
                        )
                        for fp_idx, fp in enumerate(im_list)
                    ]
                [task.get() for task in tasks]

    def run_on_h5(self, folder, target_folder):

        im_list = np.sort(glob(os.path.join(folder, '*.h5')))

        [normalize_h5_with_quantiles(fp, target_folder, self.q_lower, self.q_upper, self.quantile) for fp in im_list]


if __name__ == '__main__':

    # results_folder = '/home/hennies/Desktop/rachs_ugly_image_normalized_d40e2'
    # if not os.path.exists(results_folder):
    #     os.mkdir(results_folder)
    #
    # # from skimage.io import imread
    # # ref_im = imread('/home/hennies/Desktop/rachs_ugly_image.tif')
    # # ref_mask = imread('/home/hennies/Desktop/rachs_ugly_mask.tif').astype(bool)
    # ref_im = imread('/home/hennies/Desktop/rachs_ugly_image.tif')
    # ref_mask = imread('/home/hennies/Desktop/rachs_ugly_mask_d40e2.tif').astype(bool)
    # qn = QuantileNormalizer(ref_im, 0.1, ref_mask)
    # qn.run_on_tif_stack('/home/hennies/Desktop/rachs_ugly_image/',
    #                     results_folder,
    #                     mask_folder='/home/hennies/Desktop/rachs_ugly_mask_d40e2/')

    # results_folder = '/home/hennies/Desktop/rachs_ugly_image_normalized_no_mask'
    # if not os.path.exists(results_folder):
    #     os.mkdir(results_folder)
    #
    # ref_im = imread('/home/hennies/Desktop/rachs_ugly_image/slice_0200.tif')
    # qn = QuantileNormalizer(ref_im, 0.1)
    # qn.run_on_tif_stack('/home/hennies/Desktop/rachs_ugly_image/',
    #                     results_folder)

    # results_folder = '/home/hennies/Desktop/rachs_ugly_image_hq_normalized_no_mask_0.07'
    # if not os.path.exists(results_folder):
    #     os.mkdir(results_folder)
    #
    # ref_im = imread('/home/hennies/Desktop/rachs_ugly_image_hq.tif')
    # qn = QuantileNormalizer(ref_im, 0.07)
    # qn.run_on_tif_stack('/home/hennies/Desktop/rachs_ugly_image_hq/',
    #                     results_folder)

    results_folder = '/home/hennies/Desktop/rachs_ugly_image_hq_normalized_mask'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # from skimage.io import imread
    # ref_im = imread('/home/hennies/Desktop/rachs_ugly_image.tif')
    # ref_mask = imread('/home/hennies/Desktop/rachs_ugly_mask.tif').astype(bool)
    ref_im = imread('/home/hennies/Desktop/rachs_ugly_image_hq.tif')
    ref_mask = imread('/home/hennies/Desktop/rachs_ugly_image_hq_mask.tif').astype(bool)
    qn = QuantileNormalizer(ref_im, 0.1, ref_mask)
    qn.run_on_tif_stack('/home/hennies/Desktop/rachs_ugly_image_hq/',
                        results_folder,
                        mask_folder='/home/hennies/Desktop/rachs_ugly_image_hq_mask/')

