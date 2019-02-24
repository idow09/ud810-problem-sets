import cv2
import numpy as np


def hough_lines_draw(img, outfile, peaks, rho, theta):
    """
    Draw lines found in an image using Hough transform.

    img: Image on top of which to draw lines
    outfile: Output image filename to save plot as
    peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    rho: Vector of rho values, in pixels
    theta: Vector of theta values, in degrees
    """
    for peak in peaks:
        r, t = rho[peak[0]], theta[peak[1]]
        a = np.cos(np.deg2rad(t))
        b = np.sin(np.deg2rad(t))
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        if len(img.shape) is not 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite(outfile, img)
