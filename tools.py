
import numpy as np
import glob
import os

from tifffile import imread, imsave
import pickle

from multiprocessing import Pool

from scipy.ndimage.interpolation import geometric_transform


def _transform_from_field(idx, im_filepath, field_filepath,
                          target_folder, im_list_len,
                          field_xy_range, im_xy_range):

    def _shift_func(out_coords):
        fx = field[1]
        fy = field[0]

        # out_coords[0] += fx[out_coords]
        # out_coords[1] += fy[out_coords]
        return int(out_coords[0] + fx[out_coords]), int(out_coords[1] + fy[out_coords])

    print('idx = {}/{}'.format(idx, im_list_len - 1))

    im = imread(im_filepath)[im_xy_range]
    with open(field_filepath, mode='rb') as f:
        field = list(pickle.load(f))
    field[0] = field[0][field_xy_range]
    field[1] = field[1][field_xy_range]

    # Shifting the pixels
    result = geometric_transform(im.astype('float32'), _shift_func)

    # Write the result
    filename = os.path.split(im_filepath)[1]
    imsave(os.path.join(target_folder, filename), result.astype(im.dtype))


def transform_dataset_from_field(
        source_folder, target_folder,
        field_source_folder,
        field_z_range=np.s_[:],
        im_z_range=np.s_[:],
        field_xy_range=np.s_[:],
        im_xy_range=np.s_[:],
        n_workers=1
):

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    field_list = np.array(sorted(glob.glob(os.path.join(field_source_folder, '*.pkl'))))[field_z_range]
    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[im_z_range]
    
    assert len(im_list) == len(field_list)

    if n_workers == 1:

        for idx, im_filepath in enumerate(im_list):

            _transform_from_field(
                idx, im_filepath, field_list[idx], target_folder, len(im_list),
                field_xy_range, im_xy_range
            )

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    _transform_from_field, (
                        idx, im_filepath, field_list[idx], target_folder, len(im_list),
                        field_xy_range, im_xy_range
                    )
                )
                for idx, im_filepath in enumerate(im_list)
            ]

            [task.get() for task in tasks]


if __name__ == '__main__':
    
    source = '/g/schwab/hennies/datasets/connectomics/cremi/sample_a_neuron_ids_sliced'
    target = '/g/schwab/hennies/datasets/connectomics/cremi/sample_a_neuron_ids_sliced_realigned'
    field_source = '/g/schwab/hennies/phd_project/image_analysis/alignment/template_align/e181121_00_amst_median_3_store_field_ds_cremi_a/field'
    
    transform_dataset_from_field(
        source, target, field_source,
        field_z_range=np.s_[38:163],
        field_xy_range=np.s_[910:2160, 910:2160],
        n_workers=16
    )
