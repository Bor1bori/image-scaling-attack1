import core
import numpy as np
import cv2 as cv


def strong_attack(scale_func, source_img, target_img):
    height_s, width_s, _ = source_img.shape
    height_t, width_t, _ = target_img.shape
    print(source_img.shape, target_img.shape)
    coefficient_left, coefficient_right = core.get_coefficient(height_s, width_s, height_t, width_t)

    intermediate_source_img = np.array([np.dot(source_img[:, :, i], coefficient_right) for i in range(3)]).\
        transpose((1, 2, 0))
    print(intermediate_source_img.shape)
    perturbation_v = np.zeros((height_s, width_t, 3))
    for channel in range(3):
        for column in range(width_t):
            perturbation_v[:, column, channel] = core.get_perturbation_vertical(
                intermediate_source_img[:, column, channel],
                target_img[:, column, channel],
                coefficient_left,
                'min')

    intermediate_attack_img = (intermediate_source_img + perturbation_v).astype(np.dtype(np.uint32))

    perturbation_h = np.zeros((height_s, width_s, 3))
    for channel in range(3):
        for row in range(height_s):
            perturbation_h[row, :, channel] = core.get_perturbation_horizontal(
                source_img[row, :, channel],
                intermediate_attack_img[row, :, channel],
                coefficient_right,
                'min')

    attack_img = (source_img + perturbation_h).astype(np.dtype(np.float32))
    cv.imwrite('./images/result.jpg', cv.cvtColor(attack_img, cv.COLOR_RGB2BGR))
    scaled_attack_img = cv.resize(attack_img, (target_img.shape[1], target_img.shape[0]))
    cv.imwrite('./images/result_scaled.jpg', cv.cvtColor(scaled_attack_img, cv.COLOR_RGB2BGR))
