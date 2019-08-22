import os
import numpy as np
import glob
from multiprocessing import Pool
from tifffile import imread, imsave
from silx.image import sift
from vigra.filters import gaussianSmoothing
import pickle

from matplotlib import pyplot as plt


def _evaluate_alignment(im_0, im_1, areas, sigma=1.):
    print(im_0)

    im_0 = imread(im_0)
    im_1 = imread(im_1)

    if im_0.dtype == 'uint16':
        im_0 = (im_0 / (65536 / 255)).astype('uint8')
    if im_1.dtype == 'uint16':
        im_1 = (im_1 / (65536 / 255)).astype('uint8')

    # im_0 = gaussianSmoothing(im_0, sigma)
    # im_1 = gaussianSmoothing(im_1, sigma)

    offset = []

    for area in areas:

        crop_0 = im_0[area]
        crop_1 = im_1[area]

        crop_0 = gaussianSmoothing(crop_0, sigma)
        crop_1 = gaussianSmoothing(crop_1, sigma)

        try:
            # Do the alignment
            sa = sift.LinearAlign(crop_0, devicetype='GPU')
            aligned = sa.align(crop_1, return_all=True, shift_only=True)
            offset.append(aligned['offset'])
            sa = None
            aligned = None

        except TypeError:
            print('Warning: The alignment failed, appending [None, None]')
            offset.append([None, None])

    return offset


def evaluate_alignment_with_local_sift(source_folder, areas, n_workers=16, target_filepath=None, z_range=None,
                                       plot=False, sigma=1.):
    if z_range is None:
        imlist = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))
    else:
        imlist = np.array(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))[z_range]

    if n_workers == 1:
        offsets = [_evaluate_alignment(imlist[idx], imlist[idx + 1], areas, sigma=sigma) for idx in
                   range(len(imlist) - 1)]

    else:

        with Pool(processes=n_workers) as p:

            tasks = [
                p.apply_async(
                    _evaluate_alignment,
                    (imlist[idx], imlist[idx + 1], areas),
                    {'sigma': sigma}
                )
                for idx in range(len(imlist) - 1)
            ]

            offsets = [task.get() for task in tasks]

    offsets = np.array(offsets)

    if plot:
        for idx in range(offsets.shape[2]):
            offset_x = np.sqrt(np.power(offsets[:, 0, idx], 2) + np.power(offsets[:, 1, idx], 2))

            plt.plot(offset_x)

        plt.show()

    if target_filepath is not None:

        if not os.path.exists(os.path.split(target_filepath)[0]):
            print('Making directory {}'.format(os.path.split(target_filepath)[0]))
            os.mkdir(os.path.split(target_filepath)[0])

        print('Dumping to {}'.format(target_filepath))
        with open(target_filepath, mode='wb') as f:
            pickle.dump(offsets, f)


def plot_alignment_quality(source_filepath, multiple_plots=False,
                           pixel_size=None,
                           units=None,
                           xmin=None, ymin=None, xmax=None, ymax=None,
                           mode='offsets'
                           ):
    # if pixel_size is None:
    #     pixel_size = [1, 1]
    if units is None:
        units = ['px', 'px']

    with open(source_filepath, mode='rb') as f:
        offsets = np.array(pickle.load(f))

    offsets = offsets.astype('float64')
    colorlist = [
        'xkcd:azure',
        'xkcd:goldenrod'
    ]

    if mode == 'offsets':
        ind = 2
    elif mode == 'diffs':
        ind = 1

    # FIXME something is wrong
    ind = 1

    for idx in range(offsets.shape[ind]):
        if mode == 'offsets':
            offset_x = np.sqrt(np.power(offsets[:, idx, 0], 2) + np.power(offsets[:, idx, 1], 2))
        elif mode == 'diffs':
            offset_x = offsets[:, idx]

        if multiple_plots:
            plt.figure()

        plt.plot(
            offset_x, color=colorlist[idx]
        )
        plt.ylabel('Displacement error [{}]'.format(units[1]))
        plt.xlabel('z [{}]'.format(units[0]))

        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if xmin is not None:
            plt.xlim(xmin=xmin)
        if xmax is not None:
            plt.xlim(xmax=xmax)

        if pixel_size is not None:
            plt.xticks(plt.xticks()[0], plt.xticks()[0] * pixel_size[0])
            plt.yticks(plt.yticks()[0], plt.yticks()[0] * pixel_size[1])
            pass

        # ax = plt.gca()
        # ax.set_facecolor((0, 0, 0))


def plot_alignment_improvement(
        source_filepath, source_filepath_ref,
        pixel_size=None,
        units=None,
        xmin=None, ymin=None, xmax=None, ymax=None,
        mode='offsets'
):
    # if pixel_size is None:
    #     pixel_size = [1, 1]
    if units is None:
        units = ['px', 'px']

    with open(source_filepath, mode='rb') as f:
        offsets = np.array(pickle.load(f))

    with open(source_filepath_ref, mode='rb') as f:
        offsets_ref = np.array(pickle.load(f))

    offsets = offsets.astype('float64')
    offsets_ref = offsets_ref.astype('float64')

    colorlist = [
        'xkcd:azure',
        'xkcd:goldenrod'
    ]

    if mode == 'offsets':
        ind = 2
    elif mode == 'diffs':
        ind = 1

    # FIXME something is wrong
    ind = 1

    for idx in range(offsets.shape[ind]):
        if mode == 'offsets':
            offset_x = np.sqrt(np.power(offsets[:, idx, 0], 2) + np.power(offsets[:, idx, 1], 2))
            offset_x_ref = np.sqrt(np.power(offsets_ref[:, idx, 0], 2) + np.power(offsets_ref[:, idx, 1], 2))
        elif mode == 'diffs':
            raise NotImplementedError

        plt.plot(
            offset_x - offset_x_ref, color=colorlist[idx]
        )
        plt.ylabel('Displacement error [{}]'.format(units[1]))
        plt.xlabel('z [{}]'.format(units[0]))

        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if xmin is not None:
            plt.xlim(xmin=xmin)
        if xmax is not None:
            plt.xlim(xmax=xmax)

        if pixel_size is not None:
            plt.xticks(plt.xticks()[0], plt.xticks()[0] * pixel_size[0])
            plt.yticks(plt.yticks()[0], plt.yticks()[0] * pixel_size[1])
            pass


def statistics(source_filepath, mode='offsets'):
    with open(source_filepath, mode='rb') as f:
        offsets = np.array(pickle.load(f))

    offsets = offsets.astype('float64')

    if mode == 'offsets':
        ind = 2
    elif mode == 'diffs':
        ind = 1

    # FIXME something is wrong
    ind = 1

    for idx in range(offsets.shape[ind]):
        if mode == 'offsets':
            offset_x = np.sqrt(np.power(offsets[:, idx, 0], 2) + np.power(offsets[:, idx, 1], 2))
        elif mode == 'diffs':
            offset_x = offsets[:, idx]
        offset_x = np.nan_to_num(offset_x)
        mean = np.mean(offset_x)
        median = np.median(offset_x)
        std = np.std(offset_x)
        mx = np.max(offset_x)
        mn = np.min(offset_x)

        print('---- AREA {} ----\n'.format(idx))
        print('Mean = {}'.format(mean))
        print('Median = {}'.format(median))
        print('Std = {}'.format(std))
        print('Max = {}'.format(mx))
        print('Min = {}\n'.format(mn))


if __name__ == '__main__':

    source = '/g/schwab/hennies/datasets/20140801_Hela-wt_xy5z8nm_AS/slices_part1.2a-f_inverted'
    target = '/g/schwab/hennies/phd_project/playground/e180912_00_evaluation_of_alignments_development/offsets.pkl'
    areas = [
        np.s_[210:722, 2100:2612],
        np.s_[2000:2512, 2100:2612]
    ]
    evaluate_alignment(source, areas, n_workers=1, target_filepath=target)
