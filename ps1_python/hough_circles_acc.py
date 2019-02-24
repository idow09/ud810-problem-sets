import cv2
import numpy as np


def hough_circles_acc(bw, radius):
    h = np.zeros(bw.shape)
    z = np.zeros(bw.shape)
    for (y0, x0), is_edge in np.ndenumerate(bw):
        if is_edge:
            temp = z.copy()
            cv2.circle(temp, (x0, y0), radius, 1, thickness=1)
            h += temp
    return h


def hough_circles_acc_old(bw, radius):
    h = np.zeros(bw.shape)
    xv, yv = np.meshgrid(np.arange(bw.shape[1]), np.arange(bw.shape[0]))
    for (y0, x0), is_edge in np.ndenumerate(bw):
        if is_edge:
            h[(np.abs((xv - x0) ** 2 + (yv - y0) ** 2 - radius ** 2) < 3)] += 1
    return h
