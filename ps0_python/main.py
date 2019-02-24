import numpy as np
import cv2


def cut_off_to_uint8(data):
    np.clip(data, 0, 255, out=data)
    return data.astype('uint8')


# 2 a
img1 = cv2.imread('ps0-1-a-1.png')
img1_swapped = img1.copy()
img1_swapped[:, :, [0, 2]] = img1_swapped[:, :, [2, 0]]
cv2.imwrite('ps0-2-a-1.png', img1_swapped)

# 2 b
img1_green = img1[:, :, 1]
cv2.imwrite('ps0-2-b-1.png', img1_green)

# 2 c
img1_red = img1[:, :, 0]
cv2.imwrite('ps0-2-c-1.png', img1_red)

# 3 a
img2 = cv2.imread('ps0-1-a-2.png')
img1_mono = img1_green
img2_mono = img2[:, :, 1]  # green channel
pxl_replace = img2_mono
center_2_y = pxl_replace.shape[0] // 2
center_2_x = pxl_replace.shape[1] // 2
center_1_y = img1_mono.shape[0] // 2
center_1_x = img1_mono.shape[1] // 2
pxl_replace[center_2_y - 50:center_2_y + 50, center_2_x - 50:center_2_x + 50] = img1_mono[
                                                                                center_1_y - 50:center_1_y + 50,
                                                                                center_1_x - 50:center_1_x + 50]
cv2.imwrite('ps0-3-a-1.png', pxl_replace)

# 4 a
print(img1_green.min())
print(img1_green.max())
print(img1_green.mean())
print(img1_green.std())

# 4 b
img1_mean = img1_mono.mean()
temp = (img1_mono - img1_mean) / img1_mono.std() * 10 + img1_mean
cv2.imwrite('ps0-4-b-1.png', temp)

# 4 c
shift_2_left_kernel = np.zeros((5, 5))
shift_2_left_kernel[2, 4] = 1
img1_shifted = cv2.filter2D(img1_green, -1, shift_2_left_kernel, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('ps0-4-c-1.png', img1_shifted)

# 4 d
temp = np.uint8((np.double(img1_mono) - np.double(img1_shifted) + 255) / 2)
cv2.imwrite('ps0-4-d-1.png', temp)

# 5 a
img1_green_noise = img1.copy()
gauss_noise = np.random.normal(0, 20, img1_green.shape)
img1_green_noise[:, :, 1] = cut_off_to_uint8(np.double(img1_green) + gauss_noise)
cv2.imwrite('ps0-5-a-1.png', img1_green_noise)

# 5 b
img1_blue = img1[:, :, 2]
img1_blue_noise = img1.copy()
img1_blue_noise[:, :, 2] = cut_off_to_uint8(np.double(img1_blue) + gauss_noise)
cv2.imwrite('ps0-5-b-1.png', img1_blue_noise)
