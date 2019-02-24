import cv2
import numpy as np

import time

from find_circles import find_circles
from hough_circles_acc import hough_circles_acc
from hough_lines_acc import hough_lines_acc
from hough_lines_draw import hough_lines_draw
from hough_peaks import hough_peaks

start_time = time.time()

# 1-a
img = cv2.imread('input/ps1-input0.png')
img_edges = cv2.Canny(img, 50, 150)
cv2.imwrite('output/ps1-1-a-1.png', img_edges)

# 2-a
h, theta, rho = hough_lines_acc(img_edges, 1, np.linspace(-90 + 30, 89 + 30, 180))  # defined in hough_lines_acc.py
# h, theta, rho = hough_lines_acc(img_edges, 10, np.linspace(-90 + 30, 89 + 30, 90))  # defined in hough_lines_acc.py
h_scaled = np.uint8(h / h.max() * 255)
cv2.imwrite('output/ps1-2-a-1.png', h_scaled)

# 2-b
peaks = hough_peaks(h_scaled, 10, threshold=240, n_hood_size=[5, 5])  # defined in hough_peaks.py
h_peaks = cv2.cvtColor(h_scaled, cv2.COLOR_GRAY2RGB)
for peak in peaks:
    cv2.circle(h_peaks, (peak[1], peak[0]), 1, (0, 0, 255), thickness=-1)  # circle is x, y based
cv2.imwrite('output/ps1-2-b-1.png', h_peaks)

# 2-c
hough_lines_draw(img, 'output/ps1-2-c-1.png', peaks, rho, theta)

# 2-d
print('accumulator bin sizes: rho_step = 1 pixel, theta_step = 1 degree')
print('threshold: 180')
print('neighborhood size: 5, 5')

# 3-a
img_noisy = cv2.imread('input/ps1-input0-noise.png')
img_smooth = cv2.GaussianBlur(img_noisy, (11, 11), 11, borderType=cv2.BORDER_REPLICATE)
cv2.imwrite('output/ps1-3-a-1.png', img_smooth)

# 3-b
cv2.imwrite('output/ps1-3-b-1.png', cv2.Canny(img_noisy, 50, 150))
img_noisy_edges = cv2.Canny(img_smooth, 50, 100)
cv2.imwrite('output/ps1-3-b-2.png', img_noisy_edges)

# 3-c
h_noisy, theta_noisy, rho_noisy = hough_lines_acc(img_noisy_edges, 1, np.linspace(-90 + 30, 89 + 30, 180))
h_noisy_scaled = np.uint8(h_noisy / h_noisy.max() * 255)
peaks_noisy = hough_peaks(h_noisy_scaled, 10, threshold=120, n_hood_size=[41, 41])  # defined in hough_peaks.py
h_peaks_noisy = cv2.cvtColor(h_noisy_scaled, cv2.COLOR_GRAY2RGB)
for peak in peaks_noisy:
    cv2.circle(h_peaks_noisy, (peak[1], peak[0]), 1, (0, 0, 255), thickness=-1)  # circle is x, y based
cv2.imwrite('output/ps1-3-c-1.png', h_peaks_noisy)
hough_lines_draw(img_noisy, 'output/ps1-3-c-2.png', peaks_noisy, rho_noisy, theta_noisy)

# 4-a
img = cv2.imread('input/ps1-input1.png')
img_grey = img[:, :, 2]
img_smooth = cv2.GaussianBlur(img_grey, (1, 1), 1, borderType=cv2.BORDER_REPLICATE)
cv2.imwrite('output/ps1-4-a-1.png', img_smooth)

# 4-b
img_edges = cv2.Canny(img_smooth, 400, 500)
cv2.imwrite('output/ps1-4-b-1.png', img_edges)

# 4-c
h, theta, rho = hough_lines_acc(img_edges, 1, np.linspace(-90 + 30, 89 + 30, 180))  # defined in hough_lines_acc.py
h_scaled = np.uint8(h / h.max() * 255)
peaks = hough_peaks(h_scaled, 10, threshold=200, n_hood_size=[5, 5])  # defined in hough_peaks.py
h_peaks = cv2.cvtColor(h_scaled, cv2.COLOR_GRAY2RGB)
for peak in peaks:
    cv2.circle(h_peaks, (peak[1], peak[0]), 1, (0, 0, 255), thickness=-1)  # circle is x, y based
cv2.imwrite('output/ps1-4-c-1.png', h_peaks)
hough_lines_draw(img_grey, 'output/ps1-4-c-2.png', peaks, rho, theta)

# 5-a
cv2.imwrite('output/ps1-5-a-1.png', img_smooth)
cv2.imwrite('output/ps1-5-a-2.png', img_edges)

h_circles = hough_circles_acc(img_edges, 20)
h_circles = np.uint8(h_circles / h_circles.max() * 255)
centers = hough_peaks(h_circles, 10, threshold=160, n_hood_size=[21, 21])

img_grey_circles = cv2.cvtColor(img_grey.copy(), cv2.COLOR_GRAY2RGB)
for center in centers:
    cv2.circle(img_grey_circles, (center[1], center[0]), 20, (0, 255, 0), thickness=2)

cv2.imwrite('output/beautiful.png', h_circles)
cv2.imwrite('output/ps1-5-a-3.png', img_grey_circles)

# 5-b
centers, radii = find_circles(img_edges, [20, 50])

img_circles_20_50 = cv2.cvtColor(img_grey.copy(), cv2.COLOR_GRAY2RGB)
for i, center in enumerate(centers):
    cv2.circle(img_circles_20_50, (center[1], center[0]), radii[i], (0, 255, 0), thickness=2)

cv2.imwrite('output/ps1-5-b-1.png', img_circles_20_50)

print("Program took ", time.time() - start_time, " seconds to run")
