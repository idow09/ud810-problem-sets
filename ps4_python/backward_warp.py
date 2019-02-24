import numpy as np


def backward_warp(img, trans_mtx):
    y, x = img.shape
    coords_x, coords_y = np.meshgrid(np.arange(x), np.arange(y))
    coords_z = np.ones_like(img)
    coords = np.stack((coords_x, coords_y, coords_z), axis=2)

    inv_similarity = np.linalg.pinv(trans_mtx)
    warped_coords = coords.dot(inv_similarity.T).astype(np.int)
    warped_coords_x = warped_coords[:, :, 0]
    warped_coords_y = warped_coords[:, :, 1]
    mask_x = np.logical_and(0 <= warped_coords_x, warped_coords_x < x)
    mask_y = np.logical_and(0 <= warped_coords_y, warped_coords_y < y)
    mask = np.logical_and(mask_x, mask_y)
    img_warped = np.zeros_like(img)
    img_warped[mask] = img[warped_coords_y[mask], warped_coords_x[mask]]
    return img_warped
