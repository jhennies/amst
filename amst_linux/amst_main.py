
import os
import numpy as np
import warnings

from glob import glob
from tifffile import imread, imsave
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from vigra import gaussianSmoothing
from silx.image import sift
from scipy.ndimage.interpolation import shift
from skimage.transform import downscale_local_mean, rescale
import pyelastix
from skimage.feature import register_translation
from skimage import filters


# To check which device the SIFT runs on
print('Running SIFT on ' + sift.SiftPlan(shape=(10, 10)).ctx.devices[0].name)


def default_elastix_params():

    return dict(
        # These are copied from the default affine parameter file:
        # http://elastix.bigr.nl/wiki/images/c/c5/Parameters_Affine.txt

        # The internal pixel type, used for internal computations
        # Leave to float in general.
        # NB: this is not the type of the input images! The pixel
        # type of the input images is automatically read from the
        # images themselves.
        # This setting can be changed to "short" to save some memory
        # in case of very large 3D images.
        FixedInternalImagePixelType="float",
        MovingInternalImagePixelType="float",

        # Specify whether you want to take into account the so-called
        # direction cosines of the images. Recommended: true.
        # In some cases, the direction cosines of the image are corrupt,
        # due to image format conversions for example. In that case, you
        # may want to set this option to "false".
        UseDirectionCosines=True,

        # **************** Main Components **************************

        # The following components should usually be left as they are:
        Registration="MultiResolutionRegistration",
        Interpolator="BSplineInterpolator",
        ResampleInterpolator="FinalBSplineInterpolator",
        Resampler="DefaultResampler",

        # These may be changed to Fixed/MovingSmoothingImagePyramid.
        # See the manual.
        FixedImagePyramid="FixedRecursiveImagePyramid",
        MovingImagePyramid="MovingRecursiveImagePyramid",

        # The following components are most important:
        # The optimizer AdaptiveStochasticGradientDescent (ASGD) works
        # quite ok in general. The Transform and Metric are important
        # and need to be chosen careful for each application. See manual.
        Optimizer="AdaptiveStochasticGradientDescent",
        Transform="AffineTransform",
        Metric="AdvancedMattesMutualInformation",

        # ***************** Transformation **************************

        # Scales the affine matrix elements compared to the translations, to make
        # sure they are in the same range. In general, it's best to
        # use automatic scales estimation:
        AutomaticScalesEstimation=True,

        # Automatically guess an initial translation by aligning the
        # geometric centers of the fixed and moving.
        AutomaticTransformInitialization=True,

        # Whether transforms are combined by composition or by addition.
        # In generally, Compose is the best option in most cases.
        # It does not influence the results very much.
        HowToCombineTransforms="Compose",

        # ******************* Similarity measure *********************

        # Number of grey level bins in each resolution level,
        # for the mutual information. 16 or 32 usually works fine.
        # You could also employ a hierarchical strategy:
        #(NumberOfHistogramBins 16 32 64)
        NumberOfHistogramBins=32,

        # If you use a mask, this option is important.
        # If the mask serves as region of interest, set it to false.
        # If the mask indicates which pixels are valid, then set it to true.
        # If you do not use a mask, the option doesn't matter.
        ErodeMask=False,

        # ******************** Multiresolution **********************

        # The number of resolutions. 1 Is only enough if the expected
        # deformations are small. 3 or 4 mostly works fine. For large
        # images and large deformations, 5 or 6 may even be useful.
        NumberOfResolutions=4,

        # The downsampling/blurring factors for the image pyramids.
        # By default, the images are downsampled by a factor of 2
        # compared to the next resolution.
        # So, in 2D, with 4 resolutions, the following schedule is used:
        #(ImagePyramidSchedule 8 8  4 4  2 2  1 1 )
        # And in 3D:
        #(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
        # You can specify any schedule, for example:
        #(ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
        # Make sure that the number of elements equals the number
        # of resolutions times the image dimension.

        # ******************* Optimizer ****************************

        # Maximum number of iterations in each resolution level:
        # 200-500 works usually fine for affine registration.
        # For more robustness, you may increase this to 1000-2000.
        MaximumNumberOfIterations=250,

        # The step size of the optimizer, in mm. By default the voxel size is used.
        # which usually works well. In case of unusual high-resolution images
        # (eg histology) it is necessary to increase this value a bit, to the size
        # of the "smallest visible structure" in the image:
        #(MaximumStepLength 1.0)

        # **************** Image sampling **********************

        # Number of spatial samples used to compute the mutual
        # information (and its derivative) in each iteration.
        # With an AdaptiveStochasticGradientDescent optimizer,
        # in combination with the two options below, around 2000
        # samples may already suffice.
        NumberOfSpatialSamples=2048,

        # Refresh these spatial samples in every iteration, and select
        # them randomly. See the manual for information on other sampling
        # strategies.
        NewSamplesEveryIteration=True,
        ImageSampler="Random",

        # ************* Interpolation and Resampling ****************

        # Order of B-Spline interpolation used during registration/optimisation.
        # It may improve accuracy if you set this to 3. Never use 0.
        # An order of 1 gives linear interpolation. This is in most
        # applications a good choice.
        BSplineInterpolationOrder=1,

        # Order of B-Spline interpolation used for applying the final
        # deformation.
        # 3 gives good accuracy; recommended in most cases.
        # 1 gives worse accuracy (linear interpolation)
        # 0 gives worst accuracy, but is appropriate for binary images
        # (masks, segmentations); equivalent to nearest neighbor interpolation.
        FinalBSplineInterpolationOrder=3,

        # Default pixel value for pixels that come from outside the picture:
        DefaultPixelValue=0,

        # Choose whether to generate the deformed moving image.
        # You can save some time by setting this to false, if you are
        # only interested in the final (nonrigidly) deformed moving image
        # for example.
        WriteResultImage=True,

        # The pixel type and format of the resulting deformed moving image
        ResultImagePixelType="short",
        ResultImageFormat="mhd",
    )


def optimized_elastix_params():
    """
    Returns an elastix parameter set that is optimized for the use of AMST in terms of computational time and alignment
    quality.
    """

    # Set optimized parameters
    changed_params = dict(
        # For speed up we compromise one resolution level
        NumberOfResolutions=3,
        # Still, it make sense to start down-sampling by 8 and end with no sampling
        ImagePyramidSchedule=[8, 8, 3, 3, 1, 1],
        # Increasing this yields more robustness, but takes more time
        MaximumNumberOfIterations=200,
        # For some reason turning this off really improves the result
        AutomaticScalesEstimation=False,
        # Increased step length for low resolution iterations makes it converge faster (enables smaller number of
        # resolutions and iterations, i.e. speed-up of computation)
        MaximumStepLength=[4, 2, 1],
        # Similar to the default parameter "Random", a subset of locations is selected randomly. However, subpixel
        # locations are possible in this setting. Affects alignment quality
        ImageSampler='RandomCoordinate'
    )

    # Load elastix' defaults for affine transformations
    params = default_elastix_params()
    # Adapt the ones specified above
    for param, val in changed_params.items():
        params[param] = val

    return params

def xcorr(offset_image, image):
    image = gaussianSmoothing(image, 1)
    offset_image = gaussianSmoothing(offset_image, 1)
    image = filters.sobel(image)
    offset_image = filters.sobel(offset_image)
    shift, error, diffphase = register_translation(image, offset_image, 100)
    return (shift[1], shift[0])

def default_amst_params():

    return dict(
        median_radius=7,         # radius of the median smoothing surrounding
        elastix_params=optimized_elastix_params(),
        coarse_alignment='SIFT',     # Use SIFT to get the raw data close to the template
        sift_sigma=1.6,          # Pre-smooth data before running the SIFT
        sift_downsample=(2, 2),  # Downsample the data for the SIFT (for speed-up, downsampling by 2 should not
                                 #     compromize the final result
        n_workers=8,             # Number of CPU cores allocated
        n_workers_sift=1,        # Number of threads for the SIFT step (must be 1 if run on the GPU)
        sift_devicetype='GPU',   # Run the SIFT on GPU or CPU
        compute_range=np.s_[:],  # Select a subset of the data for alignment (good for parameter testing)
        verbose=False,           # Set to True for a more detailed console output
        write_intermediates=False,  # Set to True to also write the median smoothed template and the results of the SIFT
                                 # step to disk; two folders will be created within the specified target directory that
                                 # contain this data ('refs' and 'sift')
        raw_pattern='*.tif',     # Glob pattern describing the raw file name format (default looks for any tif file)
        pre_pattern='*.tif',     # Glob pattern describing the pre-alignment file name format
                                 # (default looks for any tif file)
        compression=0            # Value between 0 (no compression) and 9 (high compression level) for the results
    )


def pre_processing_generator(
        raw_folder,
        pre_alignment_folder,
        median_radius=7,
        sift_sigma=1.6,
        sift_downsample=None,
        n_workers=8,
        compute_range=np.s_[:],
        target_folder=None,
        raw_pattern='*.tif',
        pre_pattern='*.tif',
        force_alignment=False,
        verbose=False
):
    # __________________________________________________________________________________________________________________
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

    def _prepare_data(idx, batch_idx):

        # --------------------------------------------------------------------------------------------------------------
        # Preparation of the image data

        if len(im_list_raw) > idx:
            # Check whether the result exists if a target folder is given here
            if target_folder is not None:
                if os.path.isfile(os.path.join(target_folder, os.path.split(im_list_raw[idx])[1])):
                    # Return Nones for each slice that was already computed
                    if verbose:
                        print('{} already exists, nothing to do...'.format(os.path.split(im_list_raw[idx])[1]))
                    return None, None, None, None, None
            # Load the raw data slice
            im = imread(im_list_raw[idx])
            assert im.dtype in ['uint8', 'uint16'], 'Only the data types uint8 and uint16 are supported!'
        else:
            # This happens if the number of images does not divide by the number of batches (smaller last batch)
            # Return Nones for each missing slice to fill up the batch
            if verbose:
                print('Filling up batch...')
            return None, None, None, None, None

        # Start end end indices for the median filter, idx being the current slice position in the center
        start_id = idx - median_radius
        end_id = idx + median_radius + 1

        # Load the necessary data for the z-median filtered slice
        for load_idx, slice_idx in enumerate(range(start_id, end_id)):
            load_idx += idx - batch_idx
            # print('load_idx = {}; slice_idx = {}'.format(load_idx, slice_idx))
            if 0 <= slice_idx < len(im_list_pre):
                if template_data[load_idx] is None:
                    template_data[load_idx] = imread(im_list_pre[slice_idx])
                    assert template_data[load_idx].dtype == 'uint8', 'Only 8bit template data is supported!'
            # else:
            #     print('Template data is not None')
            # else:
            #     print('slice_idx < 0')

        # Do the median smoothing to obtain one composition image at the current z-position
        ims = [x for x in template_data[idx - batch_idx: idx - batch_idx + median_radius * 2 + 1] if x is not None]
        median_z = np.median(ims, axis=0).astype('uint8')
        del ims

        # The sift doesn't like images of different size. This fixes some cases by cropping away areas from the raw data
        # that are zero and padding it with zeros to the size of the template image
        # We are assuming here that the template data slices are larger than the non-zero region of interest in the raw
        # data.
        template_shape = median_z.shape
        if im.shape != template_shape:
            try:
                bounds = _crop_zero_padding(im)
                cropped_im = im[bounds]
                t_im = np.zeros(template_shape, dtype=im.dtype)

                t_im[0:cropped_im.shape[0], 0:cropped_im.shape[1]] = cropped_im
                im = t_im
            except ValueError as e:
                # This happened when a white slice was present (no zero pixel)
                warnings.warn('Cropping zero-padding failed with ValueError: {}'.format(str(e)))

                if force_alignment:
                    print('Trying to force the alignment')
                    # Handling the case that the template dataset (i.e. median_z) is smaller than the cropped raw
                    # input (im) in at least one dimension.
                    if np.max(np.sum((cropped_im.shape, -np.array(template_shape)), axis=0) > 0):
                        print('At least in one dimension the raw data crop is larger than a template dataset slice.')
                        print('Fixing this ...')
                        t_im = np.zeros(np.max((template_shape, cropped_im.shape), axis=0), dtype=im.dtype)
                        t_im[0:cropped_im.shape[0], 0:cropped_im.shape[1]] = cropped_im
                        im = t_im
                        median_z_new = np.zeros(np.max((template_shape, cropped_im.shape), axis=0), dtype=median_z.dtype)
                        median_z_new[0: template_shape[0], 0: template_shape[1]] = median_z
                        median_z = median_z_new
                else:
                    # print('This happens if a completely black slice is present in the data. ')
                    print('Affected slice: {}'.format(im_list_raw[idx]))
                    print('Replacing with empty slice ...')
                    im = np.zeros(template_shape, dtype=im.dtype)

        # Gaussian smooth the image and the median_z for the SIFT step
        if sift_sigma is not None:
            median_z_smooth = gaussianSmoothing(median_z, sift_sigma)
            if im.dtype == 'uint8':
                im_smooth = gaussianSmoothing(im, sift_sigma)
            else:
                # vigra.gaussianSmoothing cannot handle 16 bit data
                im_smooth = gaussianSmoothing(im.astype('float32'), sift_sigma)
                # The data for the sift step does not require to be 16 bit
                im_smooth = (im_smooth / (2 ** 16) * (2 ** 8)).astype('uint8')
        else:
            median_z_smooth = median_z.copy()
            im_smooth = im.copy()
            if im_smooth.dtype == 'uint16':
                # The data for the sift step does not require to be 16 bit
                im_smooth = (im_smooth.astype('float32') / (2 ** 16) * (2 ** 8)).astype('uint8')

        # Downsample for SIFT step for speed-up of computation
        if sift_downsample is not None:
            median_z_smooth = downscale_local_mean(median_z_smooth, sift_downsample).astype('uint8')
            im_smooth = downscale_local_mean(im_smooth, sift_downsample).astype('uint8')

        # --------------------------------------------------------------------------------------------------------------
        # Return the results
        assert im.dtype in ['uint8', 'uint16']
        assert im_smooth.dtype == 'uint8'
        assert median_z.dtype == 'uint8'
        assert median_z_smooth.dtype == 'uint8'
        return im, im_smooth, median_z, median_z_smooth, template_shape
        # --------------------------------------------------------------------------------------------------------------

    # __________________________________________________________________________________________________________________
    # Main loop which yields the batches of data

    # Assert that the input folders actually exist
    assert os.path.exists(
        raw_folder), 'The raw folder does not exist. \nEnsure the raw_folder input points to the correct location.'
    assert os.path.exists(
        pre_alignment_folder), 'The folder for the pre-alignment does not exist. \nEnsure the pre_alignment_folder input points to the correct location.'

    # Find the tif files in the respective directories for the raw data and the pre-alignment
    im_list_raw = np.sort(glob(os.path.join(raw_folder, raw_pattern)))[compute_range]
    im_list_pre = np.sort(glob(os.path.join(pre_alignment_folder, pre_pattern)))[compute_range]

    # Some assertions here to ensure the folders actually contain tif files
    assert im_list_raw.size != 0, 'The raw folder does not contain *.tif files. \nEnsure the raw_folder input points to the correct location.'
    assert im_list_pre.size != 0, 'The folder for the pre-alignment does not contain *.tif files. \nEnsure the pre_alignment_folder input points to the correct location.'

    # Warning for different numbers of tif slices in raw and pre-align folders
    if len(im_list_raw) != len(im_list_pre):
        warnings.warn('Number of slices in the raw and pre-alignment folders do not match. \n'
                      'This is not necessarily an error but indicates that you might want to consider revising your inputs.')

    for batch_idx in range(0, len(im_list_raw), n_workers):

        print('batch_idx = {}'.format(batch_idx))
        # --------------------------------------------------------------------------------------------------------------
        # Generate the median smoothed template
        if verbose:
            print('Generating median smoothed templates...')
        template_data = [None] * (n_workers + 2 * median_radius)

        # Within the here called function the data is loaded and median smoothed.
        # Multi-threading to speed things up
        # The template data variable enables access to already loaded slices which were needed for median computation at
        # a previous position
        if n_workers > 1:

            with ThreadPoolExecutor(max_workers=n_workers) as tpe:

                tasks = [
                    tpe.submit(
                        _prepare_data,
                        *(idx, batch_idx)
                    )
                    for idx in range(batch_idx, batch_idx + n_workers)
                ]

            prepared_data = [task.result() for task in tasks]

        else:
            prepared_data = [
                _prepare_data(idx, batch_idx)
                for idx in range(batch_idx, batch_idx + n_workers)
            ]
        # prepared_data contains a list for every loaded slice including raw data, raw_data for the sift,
        # median z-smoothed data, and median z-smoothed data for the sift

        # This now needs to be reshaped properly
        prepared_data = np.swapaxes(np.array(prepared_data), 0, 1)
        raw_crop = prepared_data[0].tolist()
        raw_crop_smooth = prepared_data[1].tolist()
        templates_med = prepared_data[2].tolist()
        templates_med_smooth = prepared_data[3].tolist()
        final_shape = prepared_data[4].tolist()

        # Determine file names (to later save the results with the same filename
        names = [os.path.split(im_list_raw[idx])[1]
                 if len(im_list_raw) > idx else None
                 for idx in range(batch_idx, batch_idx + n_workers)]

        # Yield the batch
        yield templates_med, templates_med_smooth, raw_crop, raw_crop_smooth, names, final_shape


def _xcorr_on_pair(fixed, moving, devicetype, verbose=False):
    h, w = fixed.shape
    h2, w2 = moving.shape
    if w != w2:
        print('Images with different widths!!')
        return (0., 0.)
    if (w > 4096):
        # perform resizing to speed up
        img = rescale(fixed, 0.5)
        img2 = rescale(moving, 0.5)
        shift = xcorr(img, img2)
        offset = (shift[0] * 2, shift[1] * 2)
    else:
        offset = xcorr(fixed, moving)
    if verbose:
        print('offset = {}'.format(offset))
    return offset


def _sift_on_pair(fixed, moving, devicetype, verbose=False):
    # Initialize the SIFT
    sift_ocl = sift.SiftPlan(template=fixed, devicetype=devicetype)
    # print("Device used for calculation: ", sift_ocl.ctx.devices[0].name)

    # Compute keypoints
    keypoints_ref = sift_ocl(fixed)
    keypoints_mov = sift_ocl(moving)

    # Match keypoints
    mp = sift.MatchPlan()
    match = mp(keypoints_ref, keypoints_mov)

    # Determine offset
    if len(match) == 0:
        print('Warning: No matching keypoints found!')
        offset = (0., 0.)
    else:
        offset = (np.median(match[:, 1].x - match[:, 0].x), np.median(match[:, 1].y - match[:, 0].y))

    if verbose:
        print('offset = {}'.format(offset))

    return offset


def _displace_slice(image, offset, result_filepath=None):
    image = shift(image, -np.round([offset[1], offset[0]]))

    if result_filepath is not None:
        imsave(result_filepath, image)

    return image


def _register_with_elastix(fixed, moving,
                           transform='AffineTransform',
                           elastix_params=None,
                           name=None,
                           verbose=False
                           ):
    if name is not None and verbose:
        print('Elastix align on {}'.format(name))

    # Set the parameters. Pyelastix offers automatic and sensible defaults
    if transform == 'AffineTransform':
        params = pyelastix.get_default_params(type='AFFINE')
    else:
        params = pyelastix.get_default_params()
    # Modify the parameters as desired by input
    if params.Transform != transform:
        warnings.warn('Transform in default settings does not match selected transform!')
    for param, val in elastix_params.items():
        setattr(params, param, val)
    # Hard-coded as integers won't work
    params.ResultImagePixelType = "float"

    # The registration
    result, _ = pyelastix.register(
        np.array(moving).astype('float32'),
        np.array(fixed).astype('float32'), params,
        verbose=0
    )

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

    return result.astype(moving.dtype)


def _write_result(filepath, image, compression=0, shape=None):
    print('Writing result in {}'.format(filepath))
    if shape is None:
        imsave(filepath, image, compress=compression)
    else:
        imsave(filepath, image[0:shape[0], 0:shape[1]], compress=compression)


def amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        median_radius=7,
        elastix_params=None,
        coarse_alignment='SIFT',
        sift_sigma=None,
        sift_downsample=None,
        n_workers=8,
        n_workers_sift=1,
        sift_devicetype='GPU',
        compute_range=np.s_[:],
        verbose=False,
        write_intermediates=False,
        raw_pattern='*.tif',
        pre_pattern='*.tif',
        compression=0,
        force_alignment=False
):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    if write_intermediates:
        if not os.path.exists(os.path.join(target_folder, 'sift')):
            os.mkdir(os.path.join(target_folder, 'sift'))
        if not os.path.exists(os.path.join(target_folder, 'refs')):
            os.mkdir(os.path.join(target_folder, 'refs'))

    # The generator yields batches of data with the the number of slices equalling the specified number of cpu workers
    # The slices in each batch of data are generated in parallel
    for templates_med, templates_med_smooth, raws_crop, raws_crop_smooth, names, final_shapes in pre_processing_generator(
            raw_folder,
            pre_alignment_folder,
            median_radius=median_radius,
            sift_sigma=sift_sigma,
            sift_downsample=sift_downsample,
            n_workers=n_workers,
            compute_range=compute_range,
            target_folder=target_folder,
            raw_pattern=raw_pattern,
            pre_pattern=pre_pattern,
            force_alignment=force_alignment,
            verbose=verbose
    ):
        if coarse_alignment is not None:
            if coarse_alignment == 'SIFT':
                print('Coarse alignment by translation done with SIFT.')
                _on_pair_func = _sift_on_pair
            else:
                print('Coarse alignment by translation done with cross correlation.')
                _on_pair_func = _xcorr_on_pair
            # Run SIFT align with one thread if computation is performed on the GPU, otherwise multi-threading speeds
            # up the rather slow CPU computation
            if n_workers_sift > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as tpe:
                    tasks = [
                        tpe.submit(
                            _on_pair_func, *(templates_med_smooth[idx], raws_crop_smooth[idx], sift_devicetype, verbose)
                        )
                        if raws_crop[idx] is not None else None
                        for idx in range(len(templates_med))
                    ]
                    offsets = [task.result()
                            if task is not None else None
                            for task in tasks]
            else:
                offsets = [
                    _on_pair_func(templates_med_smooth[idx], raws_crop_smooth[idx], sift_devicetype, verbose)
                    if raws_crop[idx] is not None else None
                    for idx in range(len(templates_med))
                ]
            if sift_downsample is not None:
                offsets = [np.array(offset) * np.array(sift_downsample)
                        if offset is not None else None
                        for offset in offsets]
            del templates_med_smooth
            del raws_crop_smooth

            # Shift the batch of data in parallel
            if n_workers > 1:
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            _displace_slice, (raws_crop[idx], offsets[idx])
                        )
                        if raws_crop[idx] is not None else None
                        for idx in range(len(raws_crop))
                    ]
                    raws_crop = [task.get()
                                if task is not None else None
                                for task in tasks]
            else:
                raws_crop = [_displace_slice(raws_crop[idx], offsets[idx])
                            if raws_crop[idx] is not None else None
                            for idx in range(len(raws_crop))]

        if write_intermediates:
            # Write the sift images in parallel
            if n_workers > 1:
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            _write_result, (
                                os.path.join(target_folder, 'sift', names[idx]), raws_crop[idx]
                            )
                        )
                        if raws_crop[idx] is not None else None
                        for idx in range(len(raws_crop))
                    ]
                    [task.get()
                    if task is not None else None
                    for task in tasks]
            else:
                [_write_result(os.path.join(target_folder, 'sift', names[idx]), raws_crop[idx])
                if raws_crop[idx] is not None else None
                for idx in range(len(raws_crop))
                ]
            # Write template images in parallel
            if n_workers > 1:
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            _write_result, (
                                os.path.join(target_folder, 'refs', names[idx]), templates_med[idx]
                            )
                        )
                        if raws_crop[idx] is not None else None
                        for idx in range(len(raws_crop))
                    ]
                    [task.get()
                    if task is not None else None
                    for task in tasks]
            else:
                [_write_result(os.path.join(target_folder, 'refs', names[idx]), templates_med[idx])
                if raws_crop[idx] is not None else None
                for idx in range(len(raws_crop))
                ]

        # Register with ELASTIX with one thread, as it is parallelized internally
        raws_crop = [
            _register_with_elastix(
                templates_med[idx], raws_crop[idx], name=names[idx],
                elastix_params=elastix_params, verbose=verbose
            )
            if raws_crop[idx] is not None else None
            for idx in range(len(raws_crop))
        ]
        del templates_med

        # Write the results in parallel
        if n_workers > 1:
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        _write_result, (
                            os.path.join(target_folder, names[idx]), raws_crop[idx], compression
                        )
                    )
                    if raws_crop[idx] is not None else None
                    for idx in range(len(raws_crop))
                ]
                [task.get()
                if task is not None else None
                for task in tasks]
        else:
            [_write_result(os.path.join(target_folder, names[idx]), raws_crop[idx], compression, final_shapes[idx])
            if raws_crop[idx] is not None else None
            for idx in range(len(raws_crop))
            ]

    

if __name__ == '__main__':

    import time
    start = time.time()

    experiment_name = 'amst_191119_00_implementation_test'
    project_name = 'amst_191119_00_implementation_tests'

    raw = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/raw_8bit'
    pre = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/template_match_aligned/'
    target = os.path.join(
        '/data/phd_project/image_analysis/alignment/fast_amst/',
        project_name,
        experiment_name
    )

    params = default_amst_params()

    params['n_workers'] = 12
    # For debugging
    params['compute_range'] = np.s_[417:687]

    amst_align(
        raw_folder=raw,
        pre_alignment_folder=pre,
        target_folder=target,
        verbose=False,
        **params
    )

    end = time.time()
    print('Time elapsed: {}s'.format(end - start))
