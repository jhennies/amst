
import os
import glob
import numpy as np
from multiprocessing import Pool
from tifffile import imread, imsave
from vigra.filters import gaussianSmoothing
import warnings


def median_z(target_folder, im_list, idx, radius):
    """
    Computes the z-median smoothed image slice at position idx

    :param target_folder: Where the result will be stored
    :param im_list: List of image file locations
    :param idx: The current position
    :param radius: Radius of the median smoothing
    :return:
    """

    filename = os.path.split(im_list[idx])[1]

    # Return if the target file already exists
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('median_z: {} exists, nothing to do'.format(filename))
        return

    print('median_z on slice {}'.format(idx))

    # Load necessary images
    start_id = idx - radius
    # Cope with the boundaries
    if start_id < 0:
        start_id = 0
    end_id = idx + radius + 1
    if end_id > len(im_list):
        end_id = len(im_list)

    ims = []
    for load_id in range(start_id, end_id):
        # print('load_id={}'.format(load_id))
        ims.append(imread(im_list[load_id]))

    # Make composite
    composite = np.median(ims, axis=0).astype(ims[0].dtype)

    # Save result
    imsave(os.path.join(target_folder, filename), data=composite)


def smoothing_and_median_z(target_folder, im_list, idx, sigma, median_radius):
    """
    Same as median_z with an additional gaussian smoothing. So far I did not see any improvement with this, so median_z is
    the recommended function.

    :param target_folder:
    :param im_list:
    :param idx:
    :param sigma:
    :param median_radius:
    :return:
    """

    filename = os.path.split(im_list[idx])[1]

    # Return if the target file already exists
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('smoothing_and_median_z: {} exists, nothing to do'.format(filename))
        return

    print('smoothing_and_median_z on slice {}'.format(idx))

    # Load necessary images
    start_id = idx - median_radius
    if start_id < 0:
        start_id = 0
    end_id = idx + median_radius + 1
    if end_id > len(im_list):
        end_id = len(im_list)

    ims = []
    for load_id in range(start_id, end_id):
        # print('load_id={}'.format(load_id))
        ims.append(gaussianSmoothing(imread(im_list[load_id]), sigma).astype('uint8'))

    # Make composite
    composite = np.median(ims, axis=0).astype(ims[0].dtype)

    # Save result
    imsave(os.path.join(target_folder, filename), data=composite)


def template_function_wrapper(func, source_folder, target_folder, *args, n_workers, source_range=np.s_[:], **kwargs):

    print('Computing {} with n_workers={}'.format(func, n_workers))
    print('args = {}'.format(args))
    print('kwargs = {}'.format(kwargs))

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[source_range]

    if n_workers == 1:

        for idx in range(len(im_list)):
            func(
                target_folder, im_list, idx, *args, **kwargs
            )

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    func, (
                        target_folder, im_list, idx, *args
                    ),
                    kwargs
                )
                for idx in range(len(im_list))
            ]

            [task.get() for task in tasks]


def defaults(func):
    """
    Implement default parameters here.
    FIXME: I am pretty sure there is a more elegant solution to this.

    :param func:
    :return:
    """
    if func == median_z:
        return (7,), {}
    if func == smoothing_and_median_z:
        return (2., 7), {}

    # Return empty if nothing is implemented
    warnings.warn('Default parameters for this function are not implemented. Returning empty paramenters and hoping for the best...')
    return (), {}
