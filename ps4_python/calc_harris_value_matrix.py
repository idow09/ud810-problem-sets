import cv2
import numpy as np


def calc_harris_value_matrix(img, alpha=0.04):
    grad_x, grad_y = grad(img)
    i_x_2 = cv2.GaussianBlur(grad_x ** 2, (5, 5), 0)
    i_y_2 = cv2.GaussianBlur(grad_y ** 2, (5, 5), 0)
    i_x_y = cv2.GaussianBlur(grad_x * grad_y, (5, 5), 0)

    r = np.empty(img.shape)
    for (i, j), _ in np.ndenumerate(r):
        moment_list = [i_x_2[i, j], i_x_y[i, j], i_x_y[i, j], i_y_2[i, j]]
        moment = np.array(moment_list).reshape(2, 2)
        r[i, j] = cv2.determinant(moment) - alpha * cv2.trace(moment)[0] ** 2
    return r


def grad(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return grad_x, grad_y
