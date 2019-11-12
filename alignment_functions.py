
import os
import numpy as np
import glob
from multiprocessing import Pool
from tifffile import imread, imsave
from scipy.ndimage.interpolation import shift
import pickle

import gc

from matplotlib import pyplot as plt

import pyelastix

from silx.image import sift
import warnings
import vigra


def _register_with_elastix(moving, ref,
                           transform,
                           number_of_resolutions,
                           maximum_number_of_iterations,
                           final_grid_spacing_in_physical_units,
                           image_pyramid_schedule
                           ):

    # Set the parameters. Pyelastix offers automatic and sensible defaults
    if transform == 'AffineTransform':
        params = pyelastix.get_default_params(type='AFFINE')
    else:
        params = pyelastix.get_default_params()
    # Modify the parameters as desired by input
    params.Transform = transform
    if number_of_resolutions is not None:
        params.NumberOfResolutions = number_of_resolutions
    if maximum_number_of_iterations is not None:
        params.MaximumNumberOfIterations = maximum_number_of_iterations
    if final_grid_spacing_in_physical_units is not None:
        params.FinalGridSpacingInPhysicalUnits = final_grid_spacing_in_physical_units
    if image_pyramid_schedule is not None:
        params.ImagePyramidSchedule = image_pyramid_schedule
    # Hard-coded as integers won't work
    params.ResultImagePixelType = "float"

    # The registration
    result, field = pyelastix.register(moving.astype('float32'),
                                       ref.astype('float32'), params,
                                       verbose=0)

    # The result is read only when it comes out of pyelastix -- copying and replacing fixes that
    result = result.copy()

    # Getting back the input datatype
    if moving.dtype == 'uint8':
        result[result < 0] = 0
        result[result > 255] = 255
    elif moving.dtype == 'uint16':
        result[result < 0] = 0
        result[result > 65535] = 65535
    else:
        raise NotImplementedError

    return result.astype(moving.dtype), field


# Helper functions
def _crop_zero_padding(dat):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    # generate bounds
    bounds = np.s_[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
                   top_left[1]:bottom_right[1] + 1]  # inclusive
    return bounds


def elastix_align_advanced(target_folder, im_filepath, ref_im_filepath,
                           connected_components=False,
                           transform='AffineTransform',
                           save_field=None,
                           background_value=0,
                           invert_for_align=False,
                           number_of_resolutions=None,
                           maximum_number_of_iterations=None,
                           final_grid_spacing_in_physical_units=None,
                           image_pyramid_schedule=None,
                           mode='no_crop'
                           ):
    """
    Performs a registration on two images using the Elastix toolkit

    :param target_folder: Where the result is saved
    :param im_filepath: The image that will be registered
    :param ref_im_filepath: The reference image
    :param connected_components: If true the image is split into ROIs by connected component analysis
    :param save_field: If this is a file path the transformation field is saved to disk
    :param background_value: Value of the background outside the regions of interest
    :param invert_for_align: Invert the image data before alignment
        Note that the parameter background_value has to match the background in the INVERTED image

    Parameters from Elastix (Refer to Elastix documentation):
    :param transform:
    :param number_of_resolutions:
    :param maximum_number_of_iterations:
    :param final_grid_spacing_in_physical_units:
    :param image_pyramid_schedule

    :param mode:
        'crop_roi': Both reference and moving image are cropped to the roi, only works if result is already close
        'no_crop': No cropping

    :return:
    """

    # >>>>>>>>>>>>>>>>>>>>>>>
    # General initializations

    if not connected_components:
        if mode == 'crop_roi':
            warnings.warn("mode='crop_roi' only implemented for connected_components=True, setting mode to 'no_crop'")
        mode = 'no_crop'

    # Getting the filename of the input image
    filename = os.path.split(im_filepath)[1]

    # Skip this file if the result is already there
    if os.path.isfile(os.path.join(target_folder, filename)):
        print('_elastix_align: {} exists, nothing to do'.format(filename))
        return

    print('_elastix_align on {}'.format(filename))

    # Load images. From now on the image list is always [moving, reference]
    ims = [
        imread(im_filepath),
        imread(ref_im_filepath)
    ]

    # Handle multiple datatypes of image and reference
    if ims[0].dtype != ims[1].dtype:
        if ims[0].dtype == 'uint16' and ims[1].dtype == 'uint8':
            ims[1] = ims[1].astype('uint16') * 255
        else:
            raise NotImplementedError

    # <<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>
    # Some helper functions

    # Invert images
    def _invert_image(im):
        if im is not None:
            return -im + im.max()
        else:
            return None

    im_max = ims[0].max()
    if invert_for_align:
        for idx, im in enumerate(ims):
            ims[idx] = _invert_image(im)

    # Split images up into connected components
    def _connected_components(moving, ref):

        inpt = np.zeros(moving.shape, dtype='float32')
        inpt[
            vigra.filters.gaussianSmoothing(moving.astype('float32'), 3) != background_value
            ] = 1
        con_comps = vigra.analysis.labelMultiArrayWithBackground(inpt, background_value=0)

        cpnts = []
        bnds = []
        for lbl_id in np.unique(con_comps)[1:]:

            if mode == 'crop_roi':

                bnds.append(
                    _crop_zero_padding(con_comps == lbl_id)
                )
                moving_crop = moving[bnds[-1]]
                moving_crop[con_comps[bnds[-1]] != lbl_id] = background_value

                ref_crop = ref[bnds[-1]]
                ref_crop[con_comps[bnds[-1]] != lbl_id] = background_value

            elif mode == 'no_crop':

                moving_crop = moving.copy()
                moving_crop[con_comps != lbl_id] = background_value

                ref_crop = ref.copy()
                ref_crop[con_comps != lbl_id] = background_value

            else:
                raise NotImplementedError
            cpnts.append(
                [
                    moving_crop, ref_crop
                ]
            )

        return cpnts, bnds

    # <<<<<<<<<<<<<<<<<<<<<<<<

    # Components is organized just as ims: [[moving, reference], ...]
    if connected_components:
        # Determine the connected components
        components, bounds = _connected_components(*ims)
    else:
        # Treat the image as one connected component
        components = [ims]
        bounds = []

    # Pre-allocate a result image
    result = np.ones(ims[1].shape, dtype=ims[0].dtype) * background_value

    print('Processing {} connected components'.format(len(components)))
    for idx, component in enumerate(components):
        print('Component {} of {}'.format(idx + 1, len(components)))

        # Align the image
        aligned, field = _register_with_elastix(component[0], component[1],
                                                transform,
                                                number_of_resolutions,
                                                maximum_number_of_iterations,
                                                final_grid_spacing_in_physical_units,
                                                image_pyramid_schedule)

        # Write results back into a result image
        if mode == 'crop_roi':
            result[bounds[idx]][aligned != background_value] = aligned[aligned != background_value]
        elif mode == 'no_crop':
            result[aligned != background_value] = aligned[aligned != background_value]
        else:
            raise NotImplementedError

    if invert_for_align:
        result = -result + im_max

    # Save the result
    imsave(os.path.join(target_folder, filename), result)

    # Save the transformations
    if save_field is not None:
        if not os.path.exists(save_field):
            os.mkdir(save_field)
        filename = os.path.splitext(filename)[0]
        with open(os.path.join(save_field, '{}.pkl'.format(filename)), mode='wb') as f:
            pickle.dump(field, f)


def sift_align(target_folder, im_filepath, ref_im_filepath, shift_only=True, subpixel_displacement=True, devicetype='CPU'):
    """
    This function is integrated to get the raw data close to the template with translations only and then run the
    elastix alignment on top.
    In theory this could also be used for affine transformations.

    :param target_folder: Where the result is saved
    :param im_filepath: The image that will be registered
    :param ref_im_filepath: The reference image
    :param shift_only:
        True: Translations only
        False: Affine transformations
    :param subpixel_displacement:
    :return:
    """

    filename = os.path.split(im_filepath)[1]

    if os.path.isfile(os.path.join(target_folder, filename)):
        print('_sift_align: {} exists, nothing to do'.format(filename))
        return

    print('_sift_align on {}'.format(filename))

    # Load images
    im = imread(im_filepath)
    ref_im = imread(ref_im_filepath)

    # The sift doesn't like images of different size. This fixes some cases by cropping away areas that are zero
    if im.shape != ref_im.shape:
        try:
            bounds = _crop_zero_padding(im)
            cropped_im = im[bounds]
            t_im = np.zeros(ref_im.shape, dtype=im.dtype)

            t_im[0:cropped_im.shape[0], 0:cropped_im.shape[1]] = cropped_im
            im = t_im
        except ValueError as e:
            # This happened when a white slice was present (no zero pixel)
            warnings.warn('Cropping zero-padding failed with ValueError: {}'.format(str(e)))
            
    # Align the image
    sa = sift.LinearAlign(ref_im, devicetype=devicetype)

    try:
        aligned = sa.align(im, return_all=True, orsa=False, shift_only=shift_only)
    except Exception as e:
        warnings.warn('Initializing SIFT failed with message: {}'.format(str(e)))
        aligned = None
    
    if aligned is not None:
        # The returned result is subpixel displaced
        if subpixel_displacement:
            result = aligned['result']
        else:
            # If we don't want subpixel displacement we can use the offset to shift the image
            # This should only be done when shift_only=True
            # FIXME this should be directly integrated into the sift 
            result = shift(im, -np.round(aligned['offset']))
    else:
        # This happens when sift failed
        result = im

    del sa

    # Write result
    imsave(os.path.join(target_folder, filename), result.astype(im.dtype))


def alignment_function_wrapper(func, source_folder, ref_source_folder, target_folder,
                               alignment_params,
                               n_workers=1, source_range=np.s_[:], ref_range=np.s_[:]):
    """
    Wrapper around the above alignment functions to run (parallelized) over a dataset.

    :param func: the alignment function which will be called
    :param source_folder: The folder from which to take the moving images
    :param ref_source_folder: The folder from which to take the fixed images
    :param target_folder: The folder to store the results
    :param alignment_params: Keyword arguments for the alignment function
    :param n_workers: For multiprocessing
        n_workers == 1: normal function call
        n_workers > 1: multiprocessing
    :param source_range: for debugging to select a subset of the moving images
    :param ref_range: for debugging to select a subset of the fixed images
    :return:
    """

    print('Computing {} with n_workers={}'.format(func, n_workers))
    print('alignment_params = {}'.format(alignment_params))

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[source_range]
    ref_im_list = np.array(sorted(glob.glob(os.path.join(ref_source_folder, '*.tif'))))[ref_range]

    if n_workers == 1:

        print('Running with one worker...')
        for idx in range(len(im_list)):
            func(
                target_folder, im_list[idx], ref_im_list[idx], **alignment_params
            )

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    func, (
                        target_folder, im_list[idx], ref_im_list[idx]
                    ),
                    alignment_params
                )
                for idx in range(len(im_list))
            ]

            [task.get() for task in tasks]


def alignment_defaults(func):
    """
    Returns default parameter sets for alignment functions.

    :param func: the function for which to return the default parameter set
    :return: the parameter set as dictionary
    """

    if func == sift_align:
        return dict(
            shift_only=False,
            subpixel_displacement=True,
            devicetype='GPU'
        )
    elif func == elastix_align_advanced:
        return dict(
            connected_components=False,
            transform='AffineTransform',
            save_field=None,
            background_value=0,
            invert_for_align=False,
            number_of_resolutions=4,
            maximum_number_of_iterations=500,
            mode='no_crop'
        )

    # Return empty if nothing is implemented
    warnings.warn('Default parameters for this function are not implemented. Returning empty paramenters and hoping for the best...')
    return dict()


