
# The code in this file is adapted from almarklein's https://github.com/almarklein/pyelastix
# The adaptations enable a use of an image mask (-fMask and -mMask arguments of elastix comand line inputs)
# TODO Fork pyelastix repo and make pull request

import numpy as np
import os

from pyelastix import _write_image_data, get_tempdir, _clear_temp_dir, _compile_params, Parameters
from pyelastix import _write_parameter_file, get_elastix_exes, _system3, _read_image_data


def _get_image_paths(im1, im2, mask):
    """ If the images are paths to a file, checks whether the file exist
    and return the paths. If the images are numpy arrays, writes them
    to disk and returns the paths of the new files.
    """

    paths = []
    for im in [im1, im2, mask]:
        if im is None:
            # Groupwise registration: only one image (ndim+1 dimensions)
            paths.append(paths[0])
            continue

        if isinstance(im, str):
            # Given a location
            if os.path.isfile(im1):
                paths.append(im)
            else:
                raise ValueError('Image location does not exist.')

        elif isinstance(im, np.ndarray):
            # Given a numpy array
            id = len(paths) + 1
            p = _write_image_data(im, id)
            paths.append(p)

        else:
            # Given something else ...
            raise ValueError('Invalid input image.')

    # Done
    return tuple(paths)


def register_with_mask(im1, im2, mask, params, exact_params=False, verbose=1):
    """ register(im1, im2, params, exact_params=False, verbose=1)

    Perform the registration of `im1` to `im2`, using the given
    parameters. Returns `(im1_deformed, field)`, where `field` is a
    tuple with arrays describing the deformation for each dimension
    (x-y-z order, in world units).

    Parameters:

    * im1 (ndarray or file location):
        The moving image (the one to deform).
    * im2 (ndarray or file location):
        The static (reference) image.
    * mask (ndarray or file location):
        A mask to indicate a region of interest
    * params (dict or Parameters):
        The parameters of the registration. Default parameters can be
        obtained using the `get_default_params()` method. Note that any
        parameter known to Elastix can be added to the parameter
        struct, which enables tuning the registration in great detail.
        See `get_default_params()` and the Elastix docs for more info.
    * exact_params (bool):
        If True, use the exact given parameters. If False (default)
        will process the parameters, checking for incompatible
        parameters, extending values to lists if a value needs to be
        given for each dimension.
    * verbose (int):
        Verbosity level. If 0, will not print any progress. If 1, will
        print the progress only. If 2, will print the full output
        produced by the Elastix executable. Note that error messages
        produced by Elastix will be printed regardless of the verbose
        level.
    """

    # Clear dir
    tempdir = get_tempdir()
    _clear_temp_dir()

    # Reference image
    refIm = im1
    if isinstance(im1, (tuple, list)):
        refIm = im1[0]

    # Check parameters
    if not exact_params:
        params = _compile_params(params, refIm)
    if isinstance(params, Parameters):
        params = params.as_dict()

    # Get paths of input images
    # path_im1, path_im2 = _get_image_paths(im1, im2)
    path_im1, path_im2, path_mask = _get_image_paths(im1, im2, mask)

    # Determine path of parameter file and write params
    path_params = _write_parameter_file(params)

    # Get path of trafo param file
    path_trafo_params = os.path.join(tempdir, 'TransformParameters.0.txt')

    # Register
    if True:

        # Compile command to execute
        command = [get_elastix_exes()[0],
                   '-m', path_im1, '-f', path_im2,
                   '-out', tempdir, '-p', path_params,
                   '-fMask', path_mask, '-mMask', path_mask]
        if verbose:
            print("Calling Elastix to register images ...")
        _system3(command, verbose)

        # Try and load result
        try:
            a = _read_image_data('result.0.mhd')
        except IOError as why:
            tmp = "An error occured during registration: " + str(why)
            raise RuntimeError(tmp)

    # Find deformation field
    if True:

        # Compile command to execute
        command = [get_elastix_exes()[1],
                   '-def', 'all', '-out', tempdir, '-tp', path_trafo_params]
        _system3(command, verbose)

        # Try and load result
        try:
            b = _read_image_data('deformationField.mhd')
        except IOError as why:
            tmp = "An error occured during transformation: " + str(why)
            raise RuntimeError(tmp)

    # Get deformation fields (for each image)
    if im2 is None:
        fields = [b[i] for i in range(b.shape[0])]
    else:
        fields = [b]

    # Pull apart deformation fields in multiple images
    for i in range(len(fields)):
        field = fields[i]
        if field.ndim == 2:
            field = [field[:, d] for d in range(1)]
        elif field.ndim == 3:
            field = [field[:, :, d] for d in range(2)]
        elif field.ndim == 4:
            field = [field[:, :, :, d] for d in range(3)]
        elif field.ndim == 5:
            field = [field[:, :, :, :, d] for d in range(4)]
        fields[i] = tuple(field)

    if im2 is not None:
        fields = fields[0]  # For pairwise reg, return 1 field, not a list

    # Clean and return
    _clear_temp_dir()
    return a, fields