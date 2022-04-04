
import numpy as np


"""
The code in here is taken and adapted from https://github.com/almarklein/pyelastix
"""

from pyelastix import get_tempdir, Parameters
from pyelastix import _clear_temp_dir, _compile_params
from pyelastix import _write_parameter_file, get_elastix_exes, _system3
from pyelastix import _read_image_data
from pyelastix import DTYPE_NP2ITK

import os
from shutil import copyfile


def transform(im, transformations, verbose=False):

    tempdir = get_tempdir()
    _clear_temp_dir()

    # Get paths of input images
    path_im, _ = _get_image_paths(im, None)

    # Compile command to execute
    command = [get_elastix_exes()[1],
               '-in', path_im,
               '-out', tempdir,
               '-tp', transformations]
    _system3(command, verbose)

    # Try and load result
    try:
        a = _read_image_data('result.mhd')
    except IOError as why:
        tmp = "An error occured during registration: " + str(why)
        raise RuntimeError(tmp)

    # Clean and return
    _clear_temp_dir()
    return a


def _write_image_data(im, id, name='im'):
    """ Write a numpy array to disk in the form of a .raw and .mhd file.
    The id is the image sequence number (1 or 2). Returns the path of
    the mhd file.
    """
    # im = im * (1.0/3000)  # TODO: WTF is this?
    # Create text
    lines = [
        "ObjectType = Image",
        "NDims = <ndim>",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        "CompressedData = False",
        # "TransformMatrix = <transmatrix>",
        "Offset = <origin>",
        "CenterOfRotation = <centrot>",
        "ElementSpacing = <sampling>",
        "DimSize = <shape>",
        "ElementType = <dtype>",
        "ElementDataFile = <fname>",
        ""
    ]
    text = '\n'.join(lines)

    # Determine file names
    tempdir = get_tempdir()
    fname_raw_ = f'{name}{id}.raw'
    fname_raw = os.path.join(tempdir, fname_raw_)
    fname_mhd = os.path.join(tempdir, f'{name}{id}.mhd')

    # Get shape, sampling and origin
    shape = im.shape
    if hasattr(im, 'sampling'):
        sampling = im.sampling
    else:
        sampling = [1 for _ in im.shape]
    if hasattr(im, 'origin'):
        origin = im.origin
    else:
        origin = [0 for _ in im.shape]

    # Make all shape stuff in x-y-z order and make it string
    shape = ' '.join([str(s) for s in reversed(shape)])
    sampling = ' '.join([str(s) for s in reversed(sampling)])
    origin = ' '.join([str(s) for s in reversed(origin)])

    # Get data type
    dtype_itk = DTYPE_NP2ITK.get(im.dtype.name, None)
    if dtype_itk is None:
        raise ValueError('Cannot convert data of this type: ' + str(im.dtype))

    # Set mhd text
    text = text.replace('<fname>', fname_raw_)
    text = text.replace('<ndim>', str(im.ndim))
    text = text.replace('<shape>', shape)
    text = text.replace('<sampling>', sampling)
    text = text.replace('<origin>', origin)
    text = text.replace('<dtype>', dtype_itk)
    text = text.replace('<centrot>', ' '.join(['0' for s in im.shape]))
    if im.ndim == 2:
        text = text.replace('<transmatrix>', '1 0 0 1')
    elif im.ndim == 3:
        text = text.replace('<transmatrix>', '1 0 0 0 1 0 0 0 1')
    elif im.ndim == 4:
        pass  # TODO: ???

    # Write data file
    f = open(fname_raw, 'wb')
    try:
        f.write(im.data)
    finally:
        f.close()

    # Write mhd file
    f = open(fname_mhd, 'wb')
    try:
        f.write(text.encode('utf-8'))
    finally:
        f.close()

    # Done, return path of mhd file
    return fname_mhd


def _get_image_paths(im1, im2, name='im'):
    """ If the images are paths to a file, checks whether the file exist
    and return the paths. If the images are numpy arrays, writes them
    to disk and returns the paths of the new files.
    """

    paths = []
    for im in [im1, im2]:
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
            p = _write_image_data(im, id, name=name)
            paths.append(p)

        else:
            # Given something else ...
            raise ValueError('Invalid input image.')

    # Done
    return tuple(paths)


def register(im1, im2, params, exact_params=False, save_transformations=None, mask_zeros=False, verbose=1):
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
    * save_transformations (str):
        Filepath (supply the full /path/to/*.txt) where the transformation txt will be saved
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

    # FIXME remove
    mask_zeros = True
    # Get paths of input images
    path_im1, path_im2 = _get_image_paths(im1, im2, name='im')
    if mask_zeros:
        path_mmask, path_fmask = _get_image_paths(
            (im1 > 0).astype('float32'),
            (im1 > 0).astype('float32'),
            name='mask'
        )
        params['ErodeMask'] = True

    # Determine path of parameter file and write params
    path_params = _write_parameter_file(params)

    # Get path of trafo param file
    path_trafo_params = os.path.join(tempdir, 'TransformParameters.0.txt')

    # Register
    if True:
        # Compile command to execute
        if mask_zeros:

            command = [get_elastix_exes()[0],
                       '-m', path_im1,
                       '-f', path_im2,
                       '-fMask', path_fmask,
                       '-mMask', path_mmask,
                       '-out', tempdir,
                       '-p', path_params]

        else:
            command = [get_elastix_exes()[0],
                       '-m', path_im1,
                       '-f', path_im2,
                       '-out', tempdir,
                       '-p', path_params]
        if verbose:
            print("Calling Elastix to register images ...")
        _system3(command, verbose)

        # Try and load result
        try:
            a = _read_image_data('result.0.mhd')
        except IOError as why:
            tmp = "An error occured during registration: " + str(why)
            raise RuntimeError(tmp)

    if save_transformations is not None:
        copyfile(path_trafo_params, save_transformations)

    # # Find deformation field
    # if True:
    #     # Compile command to execute
    #     command = [get_elastix_exes()[1],
    #                '-def', 'all',
    #                '-out', tempdir,
    #                '-tp', path_trafo_params]
    #     _system3(command, verbose)
    #
    #     # Try and load result
    #     try:
    #         b = _read_image_data('deformationField.mhd')
    #     except IOError as why:
    #         tmp = "An error occured during transformation: " + str(why)
    #         raise RuntimeError(tmp)
    #
    # # Get deformation fields (for each image)
    # if im2 is None:
    #     fields = [b[i] for i in range(b.shape[0])]
    # else:
    #     fields = [b]
    #
    # # Pull apart deformation fields in multiple images
    # for i in range(len(fields)):
    #     field = fields[i]
    #     if field.ndim == 2:
    #         field = [field[:, d] for d in range(1)]
    #     elif field.ndim == 3:
    #         field = [field[:, :, d] for d in range(2)]
    #     elif field.ndim == 4:
    #         field = [field[:, :, :, d] for d in range(3)]
    #     elif field.ndim == 5:
    #         field = [field[:, :, :, :, d] for d in range(4)]
    #     fields[i] = tuple(field)
    #
    # if im2 is not None:
    #     fields = fields[0]  # For pairwise reg, return 1 field, not a list

    # Clean and return
    _clear_temp_dir()
    return a

