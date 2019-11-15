
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
import pyelastix


def pre_processing_generator(
        raw_folder,
        pre_alignment_folder,
        median_radius=7,
        sift_sigma=1.6,
        n_workers=8
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
            median_z_smooth = median_z

        # --------------------------------------------------------------------------------------------------------------
        # Return the results
        return median_z, median_z_smooth
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

        return im, im_smooth

    # __________________________________________________________________________________________________________________
    # Main loop which yields the batches of data

    im_list_raw = np.sort(glob(os.path.join(raw_folder, '*.tif')))
    im_list_pre = np.sort(glob(os.path.join(pre_alignment_folder, '*.tif')))

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
        templates_med_smooth = templates_med[1]
        templates_med = templates_med[0]

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
        raw_crop_smooth = raw_crop[1]
        raw_crop = raw_crop[0]

        # Determine file names (to later save the results with the same filename
        names = [os.path.split(im_list_raw[idx])[1] for idx in range(batch_idx, batch_idx + n_workers)]

        # Yield the batch
        yield templates_med, templates_med_smooth, raw_crop, raw_crop_smooth, names


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


def _displace_slice(image, offset):

    image = shift(image, -np.round([offset[1], offset[0]]))

    return image


def _register_with_elastix(fixed, moving,
                           transform='AffineTransform',
                           number_of_resolutions=None,
                           maximum_number_of_iterations=None,
                           final_grid_spacing_in_physical_units=None,
                           image_pyramid_schedule=None,
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
    result, _ = pyelastix.register(
        moving.astype('float32'),
        fixed.astype('float32'), params,
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
        sift_pre_align=True,
        sift_sigma=None,
        n_workers=8,
        n_workers_sift=1,
        sift_devicetype='GPU'
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
    for templates_med, templates_med_smooth, raws_crop, raws_crop_smooth, names in pre_processing_generator(
            raw_folder,
            pre_alignment_folder,
            median_radius=median_radius,
            sift_sigma=sift_sigma,
            n_workers=n_workers
    ):

        if sift_pre_align:
            # Function 2:
            #
            # Run SIFT align with one thread
            # Initialize the SIFT
            sift_ocl = sift.SiftPlan(template=templates_med[0], devicetype=sift_devicetype)
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
        raws_crop = [
            _register_with_elastix(
                templates_med[idx], raws_crop[idx], name=names[idx],
                number_of_resolutions=2,
                maximum_number_of_iterations=500,
                image_pyramid_schedule=[2, 2, 1, 1]
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
        sift_sigma=1.6,
        n_workers=12,
        n_workers_sift=1,
        sift_devicetype='GPU'
    )
