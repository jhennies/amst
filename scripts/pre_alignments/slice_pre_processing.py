
from vigra.filters import gaussianSmoothing


def preprocess_slice(image, thresh=0, sigma=1., mask_range=None):
    if mask_range is not None:
        image[image < mask_range[0]] = 0
        image[image > mask_range[1]] = 0
    if type(thresh) != list:
        thresh = [thresh, thresh]
    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]

    if sigma > 0:
        image = gaussianSmoothing(image, sigma)

    return image
