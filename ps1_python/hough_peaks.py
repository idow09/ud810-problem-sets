import numpy as np


def hough_peaks(h, numpeaks=1, threshold=None, n_hood_size=None):
    """Find peaks in a Hough accumulator array."""
    # init params
    h = h.copy()
    if threshold is None:
        threshold = 0.5 * h.max()
    if n_hood_size is None:
        n_hood_size = np.ceil(np.array(list(h.shape)) / 100) * 2 + 1  # odd values >= size(H)/50
    peaks = np.array([], np.uint8)

    for _ in range(numpeaks):
        if h.max() >= threshold:
            r, c = np.unravel_index(h.argmax(), h.shape)
            peaks = np.append(peaks, [r, c])
            # reset the neighbourhood of the current found max
            r_min = int(max(0, r - n_hood_size[0] // 2))
            r_max = int(min(h.shape[0], r + n_hood_size[0] // 2))
            c_min = int(max(0, c - n_hood_size[1] // 2))
            c_max = int(min(h.shape[1], c + n_hood_size[1] // 2))
            h[r_min:r_max, c_min:c_max] = 0

    return peaks.reshape((peaks.size // 2), 2)
