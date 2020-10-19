import numpy as np
import cv2 as cv
import cvxpy as cvx


def get_coefficient(height_s, width_s, height_t, width_t):
    source_m = np.identity(height_s) * 255
    destination_m = (cv.resize(source_m, (height_s, height_t), interpolation=cv.INTER_LINEAR)).astype(
        np.dtype(np.uint32))

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
    intermediate_source_column = intermediate_source_column.astype(np.dtype(np.float32))

    identity = np.identity(intermediate_source_column.shape[0])
    perturbation = cvx.Variable(intermediate_source_column.shape[0])
    obj_func = cvx.Minimize(cvx.quad_form(perturbation, identity)) if obj == 'min' \
        else cvx.Maximize(cvx.quad_form(perturbation, identity))
    pixel_min = np.zeros(intermediate_source_column.shape[0])
    pixel_max = np.full(intermediate_source_column.shape[0], 255)
    epsilon = np.full(target_column.shape[0], 1)
    constraints = [
        pixel_min <= intermediate_source_column + perturbation,
        intermediate_source_column + perturbation <= pixel_max,
        cvx.abs((coefficient_left @ (intermediate_source_column + perturbation)) - target_column) <= epsilon
    ]
    prob = cvx.Problem(obj_func, constraints)

    try:
        result = prob.solve()
        np_perturbation = np.array(perturbation.value)
        return np_perturbation.reshape((intermediate_source_column.shape[0]))

    except:
        try:
            prob.solve(solver=cvx.ECOS)
        except:
            print('fail to solve')
            return np.zeros((intermediate_source_column.shape[0]))
        return np.zeros((intermediate_source_column.shape[0]))


def get_perturbation_horizontal(scaled_src_row, target_row, coefficient_right, obj):
    scaled_src_row = scaled_src_row.astype(np.dtype(np.float32))

    identity = np.identity(scaled_src_row.shape[0])
    perturbation = cvx.Variable(scaled_src_row.shape[0])
    obj_func = cvx.Minimize(cvx.quad_form(perturbation, identity)) if obj == 'min' \
        else cvx.Maximize(cvx.quad_form(perturbation, identity))
    pixel_min = np.zeros(scaled_src_row.shape[0])
    pixel_max = np.full(scaled_src_row.shape[0], 255)
    epsilon = np.full(target_row.shape[0], 1)
    constraints = [
        pixel_min <= scaled_src_row + perturbation,
        scaled_src_row + perturbation <= pixel_max,
        cvx.abs(((scaled_src_row + perturbation) @ coefficient_right) - target_row) <= epsilon
    ]
    prob = cvx.Problem(obj_func, constraints)

    try:
        result = prob.solve()
        np_perturbation = np.array(perturbation.value)
        return np_perturbation.reshape((scaled_src_row.shape[0]))

    except:
        try:
            prob.solve(solver=cvx.ECOS)
        except:
            print('fail to solve')
            return np.zeros((scaled_src_row.shape[0]))
        print('fail to solve')

