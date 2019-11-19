
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
from skimage.transform import downscale_local_mean, resize
import pyelastix
import pyelastix_mod

from skimage.morphology import disk, closing


def default_affine_params():

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


def pre_processing_generator(
        raw_folder,
        pre_alignment_folder,
        median_radius=7,
        generate_fixed_mask=None,
        blur_template=None,
        sift_sigma=1.6,
        sift_downsample=None,
        n_workers=8,
        compute_range=np.s_[:]
):

    # __________________________________________________________________________________________________________________
    # Helper functions

    def _median_z(im_list, idx, batch_idx):

        # --------------------------------------------------------------------------------------------------------------
        # Compute the median smoothed template
        start_id = idx - median_radius
        end_id = idx + median_radius + 1

        for load_idx, slice_idx in enumerate(range(start_id, end_id)):
            load_idx += idx - batch_idx
            # print('load_idx = {}; slice_idx = {}'.format(load_idx, slice_idx))
            if 0 <= slice_idx < len(im_list):
                if template_data[load_idx] is None:
                    template_data[load_idx] = imread(im_list[slice_idx])
                # else:
                #     print('Template data is not None')
            # else:
            #     print('slice_idx < 0')

        # Make composite
        ims = [x for x in template_data[idx - batch_idx: idx - batch_idx + median_radius * 2 + 1] if x is not None]
        median_z = np.median(ims, axis=0).astype('uint8')
        del ims

        # Gaussian smooth the median_z for the SIFT step
        if sift_sigma is not None:
            median_z_smooth = gaussianSmoothing(median_z, sift_sigma)
        else:
            median_z_smooth = median_z.copy()
        if blur_template is not None:
            median_z = gaussianSmoothing(median_z, blur_template)

        # Generate a sensible mask for the template
        if generate_fixed_mask is not None:
            # use_smooth = generate_fixed_mask[0]
            # threshold = generate_fixed_mask[1]
            # close = generate_fixed_mask[2]
            # mask = np.zeros(median_z_smooth.shape, dtype='uint8')
            # if use_smooth:
            #     mask[median_z_smooth > threshold] = 1
            # else:
            #     mask[median_z > threshold] = 1
            # if close is not None:
            #     selem = disk(close)
            #     mask = closing(mask, selem)

            use_smooth = generate_fixed_mask[0]
            threshold = generate_fixed_mask[1]
            close = generate_fixed_mask[2]
            if use_smooth:
                t_im = downscale_local_mean(median_z_smooth, (16, 16))
            else:
                t_im = downscale_local_mean(median_z, (16, 16))
            tmask = np.zeros(t_im.shape, dtype='uint8')
            tmask[t_im > threshold] = 1
            if close is not None:
                selem = disk(close)
                tmask = closing(tmask, selem)
            mask = np.zeros(median_z.shape, 'uint8')
            mask[resize(tmask, median_z.shape) > 0] = 1

        else:
            mask = None

        # Downsample for SIFT step for speed-up of computation
        if sift_downsample is not None:
            median_z_smooth = downscale_local_mean(median_z_smooth, sift_downsample)

        # --------------------------------------------------------------------------------------------------------------
        # Return the results
        return median_z, median_z_smooth, mask
        # --------------------------------------------------------------------------------------------------------------

    def _load_raw(im_list, idx, ref_im_shape):

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

        im = imread(im_list[idx])

        # The sift doesn't like images of different size. This fixes some cases by cropping away areas that are zero
        if im.shape != ref_im_shape:
            try:
                bounds = _crop_zero_padding(im)
                cropped_im = im[bounds]
                t_im = np.zeros(ref_im_shape, dtype=im.dtype)

                t_im[0:cropped_im.shape[0], 0:cropped_im.shape[1]] = cropped_im
                im = t_im
            except ValueError as e:
                # This happened when a white slice was present (no zero pixel)
                warnings.warn('Cropping zero-padding failed with ValueError: {}'.format(str(e)))

        # Gaussian smooth the slice for the SIFT step
        if sift_sigma is not None:
            im_smooth = gaussianSmoothing(im, sift_sigma)
        else:
            im_smooth = im
        if sift_downsample is not None:
            im_smooth = downscale_local_mean(im_smooth, sift_downsample)

        return im, im_smooth

    # __________________________________________________________________________________________________________________
    # Main loop which yields the batches of data

    im_list_raw = np.sort(glob(os.path.join(raw_folder, '*.tif')))[compute_range]
    im_list_pre = np.sort(glob(os.path.join(pre_alignment_folder, '*.tif')))[compute_range]

    for batch_idx in range(0, len(im_list_raw), n_workers):

        print('batch_idx = {}'.format(batch_idx))

        # --------------------------------------------------------------------------------------------------------------
        # Generate the median smoothed template
        print('Generating median smoothed templates...')
        template_data = [None] * (n_workers + 2 * median_radius)

        # Within the here called function the data is loaded and median smoothed.
        # Multi-threading to speed things up and enable access to the template_data variable which stores the already
        # loaded slices which were needed for median computation at a previous position
        if n_workers > 1:

            with ThreadPoolExecutor(max_workers=n_workers) as tpe:

                tasks = [
                    tpe.submit(
                        _median_z,
                        *(im_list_pre, idx, batch_idx)
                    )
                    for idx in range(batch_idx, batch_idx + n_workers)
                ]

            templates_med = [task.result() for task in tasks]

        else:
            templates_med = [
                _median_z(im_list_pre, idx, batch_idx)
                for idx in range(batch_idx, batch_idx + n_workers)
            ]

        # Reshape the results
        templates_med = np.swapaxes(np.array(templates_med), 0, 1)
        masks = templates_med[2].tolist()
        templates_med_smooth = templates_med[1].tolist()
        templates_med = templates_med[0].tolist()

        # --------------------------------------------------------------------------------------------------------------
        # Load the raw data
        print('Loading raw data...')
        if n_workers > 1:

            with ThreadPoolExecutor(max_workers=n_workers) as tpe:

                tasks = [
                    tpe.submit(
                        _load_raw,
                        *(im_list_raw, idx, templates_med[0].shape)
                    )
                    for idx in range(batch_idx, batch_idx + n_workers)
                ]

            raw_crop = [task.result() for task in tasks]

        else:

            raw_crop = [
                _load_raw(im_list_raw, idx, templates_med[0].shape)
                for idx in range(batch_idx, batch_idx + n_workers)
            ]

        # Reshape the results
        raw_crop = np.swapaxes(np.array(raw_crop), 0, 1)
        raw_crop_smooth = raw_crop[1].tolist()
        raw_crop = raw_crop[0].tolist()

        # Determine file names (to later save the results with the same filename
        names = [os.path.split(im_list_raw[idx])[1] for idx in range(batch_idx, batch_idx + n_workers)]

        # Yield the batch
        yield templates_med, templates_med_smooth, raw_crop, raw_crop_smooth, masks, names


def _sift_on_pair(fixed, moving, sift_ocl, devicetype='GPU'):

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
        offset = (np.median(match[:, 1].x-match[:, 0].x), np.median(match[:, 1].y-match[:, 0].y))

    print('offset = {}'.format(offset))

    return offset


def _displace_slice(image, offset, result_filepath=None):

    image = shift(image, -np.round([offset[1], offset[0]]))

    if result_filepath is not None:
        imsave(result_filepath, image)

    return image


def _register_with_elastix(fixed, moving, mask=None,
                           transform='AffineTransform',
                           elastix_params=None,
                           name=None
                           ):

    if name is not None:
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
    if mask is not None:
        result, _ = pyelastix_mod.register_with_mask(
            np.array(moving).astype('float32'),
            np.array(fixed).astype('float32'),
            np.array(mask).astype('uint8'),
            params,
            verbose=0
        )
    else:
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


def _write_result(filepath, image):
    print('Writing result in {}'.format(filepath))
    imsave(filepath, image)


def amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        median_radius=7,
        blur_template=None,
        elastix_params=None,
        elastix_full_batch=False,
        elastix_generate_mask=None,
        sift_pre_align=True,
        sift_sigma=None,
        sift_downsample=None,
        n_workers=8,
        n_workers_sift=1,
        sift_devicetype='GPU',
        compute_range=np.s_[:]
):

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # Function 1:
    #
    # This can potentially be a generator that yields batches
    # Load batch of raw data in parallel
    # Load batch of template data in parallel
    # median smooth template data in parallel
    # Crop raw data according to template in parallel
    # Process batch in parallel
    for templates_med, templates_med_smooth, raws_crop, raws_crop_smooth, masks, names in pre_processing_generator(
            raw_folder,
            pre_alignment_folder,
            median_radius=median_radius,
            generate_fixed_mask=elastix_generate_mask,
            blur_template=blur_template,
            sift_sigma=sift_sigma,
            sift_downsample=sift_downsample,
            n_workers=n_workers,
            compute_range=compute_range
    ):

        if sift_pre_align:
            # Function 2:
            #
            # Run SIFT align with one thread
            # Initialize the SIFT
            sift_ocl = sift.SiftPlan(template=templates_med_smooth[0], devicetype=sift_devicetype)
            print("Device used for calculation: ", sift_ocl.ctx.devices[0].name)
            # Run the SIFT
            if n_workers_sift > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as tpe:
                    tasks = [
                        tpe.submit(
                            _sift_on_pair, *(templates_med_smooth[idx], raws_crop_smooth[idx], sift_ocl)
                        )
                        for idx in range(len(templates_med))
                    ]
                    offsets = [task.result() for task in tasks]
            else:
                offsets = [
                    _sift_on_pair(templates_med_smooth[idx], raws_crop_smooth[idx], sift_ocl=sift_ocl)
                    for idx in range(len(templates_med))
                ]
            if sift_downsample is not None:
                offsets = np.prod((np.array(offsets), np.array(sift_downsample)), axis=0)
            del templates_med_smooth
            del raws_crop_smooth

            # Function 3:
            #
            # Shift the batch of data in parallel
            if n_workers > 1:
                with Pool(processes=n_workers) as p:
                    tasks = [
                        p.apply_async(
                            _displace_slice, (raws_crop[idx], offsets[idx])
                        )
                        for idx in range(len(raws_crop))
                    ]
                    raws_crop = [task.get() for task in tasks]
            else:
                raws_crop = [_displace_slice(raws_crop[idx], offsets[idx]) for idx in range(len(raws_crop))]

        # # Function 4:
        # #
        # # Register with ELASTIX with one thread
        if elastix_full_batch:
            raws_crop = _register_with_elastix(templates_med, raws_crop, masks, names[0], elastix_params)
        else:
            raws_crop = [
                _register_with_elastix(
                    templates_med[idx], raws_crop[idx], mask=masks[idx], name=names[idx],
                    elastix_params=elastix_params
                )
                for idx in range(len(raws_crop))
            ]
            del templates_med

        # Function 5:
        #
        # Write the results in parallel
        if n_workers > 1:
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        _write_result, (
                            os.path.join(target_folder, names[idx]), raws_crop[idx]
                        )
                    )
                    for idx in range(len(raws_crop))
                ]
                [task.get() for task in tasks]
        else:
            [_write_result(os.path.join(target_folder, names[idx]), raws_crop[idx]) for idx in range(len(raws_crop))]


if __name__ == '__main__':

    # raw = '/g/schwab/hennies/datasets/20140801_Hela-wt_xy5z8nm_AS/20140801_hela-wt_xy5z8nm_as_full_8bit'
    # pre = '/g/schwab/hennies/datasets/20140801_Hela-wt_xy5z8nm_AS/20140801_hela-wt_xy5z8nm_as_full_8bit/template_match_aligned/'
    # target = '/g/schwab/hennies/phd_project/image_analysis/alignment/amst/amst_devel_20191115_00/'
    raw = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/raw_8bit'
    pre = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/template_match_aligned/'
    target = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/fast_amst_test/'
    # raw = '/data/datasets/20140801_hela-wt_xy5z8nm_as_full_8bit/sift/'

    amst_align(
        raw_folder=raw,
        pre_alignment_folder=pre,
        target_folder=target,
        sift_pre_align=True,
        sift_sigma=0.5,
        n_workers=12,
        n_workers_sift=1,
        sift_devicetype='GPU'
    )
