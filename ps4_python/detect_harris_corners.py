import numpy as np
from scipy.ndimage.filters import maximum_filter


def detect_harris_corners(harris_r_mtx, threshold=0.3, size=5):
    # move to positive range
    positive_r_mtx = harris_r_mtx - harris_r_mtx.min()

    # find max values (for each pixel!)
    local_max = maximum_filter(positive_r_mtx, size=size, mode='constant')

    # threshold (cut off low values)
    local_max[local_max < threshold * local_max.max()] = 0

    # find truly max original pixels
    local_max = local_max == positive_r_mtx

    return np.where(local_max)
