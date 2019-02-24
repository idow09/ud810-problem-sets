import numpy as np

from hough_circles_acc import hough_circles_acc
from hough_peaks import hough_peaks


def find_circles(bw, radius_range):
    centers = np.empty((0, 2), int)
    radii = []
    for i, rad in enumerate(np.arange(radius_range[0], radius_range[1])):
        h_circles = hough_circles_acc(bw, rad)
        # h_circles = np.uint8(h_circles / h_circles.max() * 255)
        print('--------------------------- Radius: ', rad, ' -----------------------------')
        centers_ = hough_peaks(h_circles, 5, threshold=70, n_hood_size=[21, 21])
        centers = np.concatenate((centers, centers_), axis=0)
        radii += [rad] * centers_.shape[0]

    return centers, np.array(radii)
