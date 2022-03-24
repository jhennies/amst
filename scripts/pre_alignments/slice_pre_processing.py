
from vigra.filters import gaussianSmoothing


def preprocess_slice(image, thresh=(0, 0), sigma=1., mask_range=None):

    if mask_range is not None:
        assert image.dtype == 'uint8', 'Only implemented for uint8 datasets (for now)'
        image[image < mask_range[0]] = mask_range[0]
        image[image > mask_range[1]] = mask_range[0]
        image = (image - mask_range[0]).astype('float32')
        image = (image / (mask_range[1] - mask_range[0]) * 255).astype('uint8')
    if type(thresh) != list:
        thresh = [thresh, thresh]
    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]

    if sigma > 0:
        dtype = image.dtype
        image = gaussianSmoothing(image.astype('float32'), sigma)
        image = image.astype(dtype)

    return image
