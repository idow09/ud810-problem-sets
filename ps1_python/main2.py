import cv2
import numpy as np

import time

start_time = time.time()

# 6-a
img = cv2.imread('input/ps1-input2.png')
img_grey = img[:, :, 2]
img_smooth = cv2.GaussianBlur(img_grey, (11, 11), 7, borderType=cv2.BORDER_REPLICATE)
cv2.imwrite('output/ps1-6-a-1.png', img_smooth)
img_edges = cv2.Canny(img_smooth, 25, 15)
cv2.imwrite('output/temp.png', img_edges)

# 4-b


print("Program took ", time.time() - start_time, " seconds to run")
