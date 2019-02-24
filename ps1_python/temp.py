import cv2
import numpy as np

Z = np.zeros((7, 7))
Z[2, 4] = 1
Z[3, 5] = 1
print(np.where(Z == 1))
