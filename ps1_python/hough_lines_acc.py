import numpy as np


def hough_lines_acc(bw, rho_step=1, theta=np.linspace(-90, 89, 180)):
    """
    Compute Hough accumulator array for finding lines.

    bw: Binary (black and white) image containing edge pixels
    RhoResolution (optional): Difference between successive rho values, in pixels
    Theta (optional): Vector of theta values to use, in degrees

    Please see the Matlab documentation for hough():
    http://www.mathworks.com/help/images/ref/hough.html
    Your code should imitate the Matlab implementation.

    Pay close attention to the coordinate system specified in the assignment.
    Note: Rows of h should correspond to values of rho, columns those of theta.
    """
    max_rho = np.linalg.norm(bw.shape)
    rho = np.arange(-max_rho, max_rho, step=rho_step)
    h = np.zeros((rho.size, theta.size))
    for (y, x), is_edge in np.ndenumerate(bw):
        if is_edge:
            for t_i, t in np.ndenumerate(theta):
                r = x * np.cos(np.deg2rad(t)) + y * np.sin(np.deg2rad(t))
                nearest_r_i = np.abs(rho - r).argmin()
                h[nearest_r_i, t_i] += 1
    return h, theta, rho
