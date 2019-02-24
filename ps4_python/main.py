import random
import time

import cv2
import numpy as np

from backward_warp import backward_warp
from calc_harris_value_matrix import calc_harris_value_matrix, grad
from detect_harris_corners import detect_harris_corners


def scale_to_255(arr):
    return 255 * (arr - np.min(arr)) / np.ptp(arr).astype(int)


def get_scaled_adjoined_grads(img):
    grad_x, grad_y = grad(img)
    grad_adjoined = np.concatenate((grad_x, grad_y), axis=1)
    return scale_to_255(grad_adjoined)


def detect_and_draw_harris_corners(img, harris_r_mtx, output_file):
    img_with_peaks = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    peaks_i, peaks_j = detect_harris_corners(harris_r_mtx)
    for i, j in zip(peaks_i, peaks_j):
        cv2.circle(img_with_peaks, (j, i), 1, (0, 255, 0), -1)
    cv2.imwrite(output_file, img_with_peaks)


def detect_and_draw_reach_keypoints(img):
    keypoints = extract_keypoints(img, uniform_size=False)
    return cv2.drawKeypoints(img.copy(), keypoints, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def extract_keypoints(img, uniform_size=False):
    grad_x, grad_y = grad(img)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x)
    peaks_i, peaks_j = detect_harris_corners(calc_harris_value_matrix(img))
    keypoints = []
    for i, j in zip(peaks_i, peaks_j):
        size = 1 if uniform_size else magnitude[i, j] / 500
        keypoints.append(
            cv2.KeyPoint(j, i, _size=size, _angle=180 + np.rad2deg(angle[i, j]), _octave=0))
    return keypoints


def extract_matches(img_a, img_b, match_distance_threshold=None):
    sift = cv2.xfeatures2d.SIFT_create()
    bfm = cv2.BFMatcher()
    img_a_keypoints = extract_keypoints(img_a, uniform_size=True)
    img_b_keypoints = extract_keypoints(img_b, uniform_size=True)
    _, img_a_descriptors = sift.compute(img_a, img_a_keypoints)
    _, img_b_descriptors = sift.compute(img_b, img_b_keypoints)
    matches = bfm.match(img_a_descriptors, img_b_descriptors)
    if match_distance_threshold is not None:
        matches = list(filter(lambda m: m.distance < match_distance_threshold, matches))
    return img_a_keypoints, img_b_keypoints, matches


def draw_putative_pair_image(img_a, img_b, img_a_keypoints, img_b_keypoints, matches, output_file):
    putative_pair_image = cv2.cvtColor(np.concatenate((img_a, img_b), axis=1), cv2.COLOR_GRAY2RGB)
    for match in matches:
        src, dst = get_points_from_match(match, img_a_keypoints, img_b_keypoints)
        dst = (img_a.shape[1] + dst[0], dst[1])
        color = np.random.randint(0, 255, 3).tolist()
        cv2.line(putative_pair_image, src, dst, color=color, thickness=2)
    cv2.imwrite(output_file, putative_pair_image)


def get_points_from_match(match, img_a_keypoints, img_b_keypoints):
    src = img_a_keypoints[match.queryIdx].pt
    src = (int(src[0]), int(src[1]))
    dst = img_b_keypoints[match.trainIdx].pt
    dst = (int(dst[0]), int(dst[1]))
    return src, dst


def compute_consensus_translation(img_a_keypoints, img_b_keypoints, matches, max_distance_threshold):
    consensus_sets_list = []
    for match in matches:
        consensus_set = set()
        src, dst = get_points_from_match(match, img_a_keypoints, img_b_keypoints)
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        for candidate_match in matches:
            candidate_src, candidate_dst = get_points_from_match(candidate_match, img_a_keypoints, img_b_keypoints)
            candidate_dst_estimated = (candidate_src[0] + dx, candidate_src[1] + dy)
            if distance(candidate_dst_estimated, candidate_dst) < max_distance_threshold:
                consensus_set.add(candidate_match)
        consensus_sets_list.append(consensus_set)
    consensus_sets_list.sort(key=(lambda con_set: len(con_set)), reverse=True)
    print('Translation - Matches Amount: ', len(matches))
    print('Translation - Length of biggest consensus set: ', len(consensus_sets_list[0]))
    print('Translation - Percentage of biggest consensus set from total matches: ',
          len(consensus_sets_list[0]) / len(matches) * 100)
    return list(consensus_sets_list[0])


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_consensus_similarity(img_a_keypoints, img_b_keypoints, matches, max_distance_threshold):
    consensus_sets = []
    for i in range(0, 20):
        sample_matches = random.sample(matches, 2)
        consensus_set = set()
        match1_src, match1_dst = get_points_from_match(sample_matches[0], img_a_keypoints, img_b_keypoints)
        match2_src, match2_dst = get_points_from_match(sample_matches[1], img_a_keypoints, img_b_keypoints)
        a, b, c, d = compute_similarity_transformation(match1_src, match1_dst, match2_src, match2_dst)
        for candidate_match in matches:
            candidate_src, candidate_dst = get_points_from_match(candidate_match, img_a_keypoints, img_b_keypoints)
            candidate_dst_estimated = apply_similarity_transformation(a, b, c, d, candidate_src)
            if distance(candidate_dst_estimated, candidate_dst) < max_distance_threshold:
                consensus_set.add(candidate_match)
        consensus_sets.append(consensus_set)
    consensus_sets.sort(key=(lambda con_set: len(con_set)), reverse=True)
    print('Similarity - Matches Amount: ', len(matches))
    print('Similarity - Length of biggest consensus set: ', len(consensus_sets[0]))
    print('Similarity - Percentage of biggest consensus set from total matches: ',
          len(consensus_sets[0]) / len(matches) * 100)
    return list(consensus_sets[0])


def compute_similarity_transformation(src1, dst1, src2, dst2):
    a = np.array([[src1[0], -src1[1], 1, 0],
                  [src1[1], src1[0], 0, 1],
                  [src2[0], -src2[1], 1, 0],
                  [src2[1], src2[0], 0, 1]])
    b = np.array([dst1[0], dst1[1], dst2[0], dst2[1]])
    x = np.linalg.solve(a, b)
    return x[0], x[1], x[2], x[3]


def apply_similarity_transformation(a, b, c, d, pt):
    mtx = np.array([[a, -b, c],
                    [b, a, d]])
    pt_dst = np.matmul(mtx, np.array([pt[0], pt[1], 1]))
    return pt_dst[0], pt_dst[1]


start_time = time.time()
print()

# load images
transA = cv2.imread('input/transA.jpg', cv2.IMREAD_GRAYSCALE)
transB = cv2.imread('input/transB.jpg', cv2.IMREAD_GRAYSCALE)
simA = cv2.imread('input/simA.jpg', cv2.IMREAD_GRAYSCALE)
simB = cv2.imread('input/simB.jpg', cv2.IMREAD_GRAYSCALE)

# 1-a
cv2.imwrite('output/ps4-1-a-1.png', get_scaled_adjoined_grads(transA))
cv2.imwrite('output/ps4-1-a-2.png', get_scaled_adjoined_grads(simA))

# 1-b
harris_trans_a = calc_harris_value_matrix(transA)
harris_trans_b = calc_harris_value_matrix(transB)
harris_sim_a = calc_harris_value_matrix(simA)
harris_sim_b = calc_harris_value_matrix(simB)
cv2.imwrite('output/ps4-1-b-1.png', scale_to_255(harris_trans_a))
cv2.imwrite('output/ps4-1-b-2.png', scale_to_255(harris_trans_b))
cv2.imwrite('output/ps4-1-b-3.png', scale_to_255(harris_sim_a))
cv2.imwrite('output/ps4-1-b-4.png', scale_to_255(harris_sim_b))

# 1-c
detect_and_draw_harris_corners(transA, harris_trans_a, 'output/ps4-1-c-1.png')
detect_and_draw_harris_corners(transB, harris_trans_b, 'output/ps4-1-c-2.png')
detect_and_draw_harris_corners(simA, harris_sim_a, 'output/ps4-1-c-3.png')
detect_and_draw_harris_corners(simB, harris_sim_b, 'output/ps4-1-c-4.png')

# 2-a
trans_a_with_kp = detect_and_draw_reach_keypoints(transA)
trans_b_with_kp = detect_and_draw_reach_keypoints(transB)
cv2.imwrite('output/ps4-2-a-1.png', np.concatenate((trans_a_with_kp, trans_b_with_kp), axis=1))

sim_a_with_kp = detect_and_draw_reach_keypoints(simA)
sim_b_with_kp = detect_and_draw_reach_keypoints(simB)
cv2.imwrite('output/ps4-2-a-2.png', np.concatenate((sim_a_with_kp, sim_b_with_kp), axis=1))

# 2-b
trans_a_keypoints, trans_b_keypoints, trans_matches = extract_matches(transA, transB, match_distance_threshold=110)
draw_putative_pair_image(transA, transB, trans_a_keypoints, trans_b_keypoints, trans_matches, 'output/ps4-2-b-1.png')
sim_a_keypoints, sim_b_keypoints, sim_matches = extract_matches(simA, simB, match_distance_threshold=110)
draw_putative_pair_image(simA, simB, sim_a_keypoints, sim_b_keypoints, sim_matches, 'output/ps4-2-b-2.png')

# 3-a
trans_consensus_matches = compute_consensus_translation(trans_a_keypoints, trans_b_keypoints, trans_matches,
                                                        max_distance_threshold=10)
draw_putative_pair_image(transA, transB, trans_a_keypoints, trans_b_keypoints, trans_consensus_matches,
                         'output/ps4-3-a-1.png')
print()

# 3-b
sim_consensus_matches = compute_consensus_similarity(sim_a_keypoints, sim_b_keypoints, sim_matches,
                                                     max_distance_threshold=30)
draw_putative_pair_image(simA, simB, sim_a_keypoints, sim_b_keypoints, sim_consensus_matches, 'output/ps4-3-b-1.png')

# blend it buddy
_match1_src, _match1_dst = get_points_from_match(sim_consensus_matches[0], sim_a_keypoints, sim_b_keypoints)
_match2_src, _match2_dst = get_points_from_match(sim_consensus_matches[1], sim_a_keypoints, sim_b_keypoints)
_a, _b, _c, _d = compute_similarity_transformation(_match1_src, _match1_dst, _match2_src, _match2_dst)
similarity_trans = np.array([[_a, -_b, _c],
                             [_b, _a, _d],
                             [0, 0, 1]])
warped = backward_warp(simA, similarity_trans)
cv2.imwrite('output/simA_warped.png', warped)
cv2.imwrite('output/simB_simA_transformed_blended.png', (warped * 0.5 + simB * 0.5))

print("\nProgram took ", time.time() - start_time, " seconds to run")
