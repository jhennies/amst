
import os
import numpy as np
import glob
from multiprocessing import Pool
from tifffile import imread, imsave
from scipy.ndimage.interpolation import shift
import pickle

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
                           image_pyramid_schedule,
                           automatic_transform_initialization=True,
                           automatic_scales_estimation=False,
                           ):

    # FIXME: this should not be set like this by default
    if image_pyramid_schedule is None:
        image_pyramid_schedule = [8, 8, 4, 4, 2, 2, 1, 1]

    # Set the parameters. Pyelastix offers automatic and sensible defaults
    if transform == 'AffineTransform':
        params = pyelastix.get_default_params(type='AFFINE')
    else:
        params = pyelastix.get_default_params()
    # Modify the parameters as desired by input
    params.Transform = transform
    params.NumberOfResolutions = number_of_resolutions
    params.MaximumNumberOfIterations = maximum_number_of_iterations
    params.FinalGridSpacingInPhysicalUnits = final_grid_spacing_in_physical_units
    params.AutomaticTransformInitialization = automatic_transform_initialization
    params.AutomaticScalesEstimation = automatic_scales_estimation
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
                           number_of_resolutions=4,
                           maximum_number_of_iterations=500,
                           final_grid_spacing_in_physical_units=16,
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
    :param mode:
        'crop_roi': Both reference and moving image are cropped to the roi, only works if result is already close
        'no_crop': No cropping

    :return:
    """

    # General initializations
    # >>>>>>>>>>>>>>>>>>>>>>>
    if not connected_components:
        mode = 'no_crop'

    if image_pyramid_schedule is None:
        image_pyramid_schedule = [8, 8, 4, 4, 2, 2, 1, 1]

    filename = os.path.split(im_filepath)[1]

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

    # Components is organized just as ims: [[moving, thresholded moving, reference], ...]
    if connected_components:
        components, bounds = _connected_components(*ims)
    else:
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


def sift_align(target_folder, im_filepath, ref_im_filepath, shift_only=True, subpixel_displacement=True):
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
    sa = sift.LinearAlign(ref_im, devicetype='GPU')

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

    # FIXME For some reason I now get an OUT_OF_RESOURCES error
    # Do I have to dispose the sa?
    # FIXME Does this work?
    sa.free_buffers()

    # Write result
    imsave(os.path.join(target_folder, filename), result.astype(im.dtype))


def alignment_function_wrapper(func, source_folder, ref_source_folder, target_folder,
                               *args, n_workers=1, source_range=np.s_[:], ref_range=np.s_[:], **kwargs):
    print('Computing {} with n_workers={}'.format(func, n_workers))
    print('args = {}'.format(args))
    print('kwargs = {}'.format(kwargs))

    im_list = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[source_range]
    ref_im_list = np.array(sorted(glob.glob(os.path.join(ref_source_folder, '*.tif'))))[ref_range]

    if n_workers == 1:

        for idx in range(len(im_list)):
            func(
                target_folder, im_list[idx], ref_im_list[idx], *args, **kwargs
            )

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    func, (
                        target_folder, im_list[idx], ref_im_list[idx], *args
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

    if func == sift_align:
        return (), {'shift_only': False,
                    'subpixel_displacement': True}
    elif func == elastix_align:
        return (), {'transform': 'AffineTransform',
                    'number_of_resolutions': 4}
    elif func == elastix_on_connected_components:
        return (), {'transform': 'AffineTransform',
                    'number_of_resolutions': 4,
                    'background_value': 0}

    # Return empty if nothing is implemented
    warnings.warn('Default parameters for this function are not implemented. Returning empty paramenters and hoping for the best...')
    return (), {}


