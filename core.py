import numpy as np
import cv2 as cv


def get_coefficient(height_s, width_s, height_t, width_t):
    source_m = np.identity(height_s) * 255
    destination_m = (cv.resize(source_m, (height_s, height_t), interpolation=cv.INTER_LINEAR)).astype(np.dtype(np.uint32))

    coefficient_left = destination_m / 255

    for i in range(height_t):
        coefficient_left[i, :] = coefficient_left[i, :] / coefficient_left[i, :].sum()
        assert abs(1 - coefficient_left[i, :].sum()) < 0.000001

    source_n = np.identity(width_s) * 255
    destination_n = (cv.resize(source_n, (width_t, width_s), interpolation=cv.INTER_LINEAR)).astype(np.dtype(np.uint32))
    coefficient_right = destination_n / 255
    for i in range(width_t):
        coefficient_right[:, i] = coefficient_right[:, i] / coefficient_right[:, i].sum()
        assert abs(1 - coefficient_right[:, i].sum()) < 0.000001

    return coefficient_left, coefficient_right


def get_perturbation_vertical(intermediate_source_column, target_column, coefficient_left, obj):

    return np.zeros((intermediate_source_column.shape[0]))


def get_perturbation_horizontal(scaled_src_row, target_row, coefficient_right, obj):

    return np.zeros((scaled_src_row.shape[0]))

