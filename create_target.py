#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

fractal = np.array([[[0, 0], [0, 0]], [[-1, 1], [1, 1]], [[1, -1], [1, 1]],
                    [[1, 1], [1, -1]], [[1, 1], [-1, 1]], [[-1, -1], [1, -1]],
                    [[-1, -1], [-1, 1]], [[-1, 1], [-1, -1]],
                    [[1, -1], [-1, -1]]])


def get_generator(rng=None, seed=None):
    if rng is None:
        if seed is None:
            rng = default_rng()
        else:
            rng = default_rng(seed)
    elif seed is not None:
        print(
            'Warning: the provided seed is bypassed since a RNG is also provided'
        )
    return rng


def next_step(current_table, rng):
    Y, X = current_table.shape
    sign = fractal[current_table].transpose(0, 2, 1, 3).reshape(2 * Y, 2 * X)
    new_table = sign * rng.integers(low=1, high=5, size=sign.shape)
    return new_table


def create_targets(n_scales, rng=None, seed=None):
    rng = get_generator(rng, seed)
    init_sign = -1 if rng.random() < .5 else 1
    table = init_sign * rng.integers(low=1, high=5, size=((1, 1)))
    targets = [table]
    for i_scale in range(1, n_scales):
        targets.append(next_step(targets[-1], rng))
    return targets


def target2img(target, fine_checker_size=16, coarse_checker_size=16, repeat=1):
    Y, X = target.shape
    target_center = (fractal[target].transpose(0, 2, 1, 3).reshape(
        2 * Y, 2 * X) > 0).astype(np.uint8)
    Y_ori, X_ori = target_center.shape
    total_checker_size = coarse_checker_size + fine_checker_size
    Y, X = Y_ori + 2 * total_checker_size, X_ori + 2 * total_checker_size

    # initialize image
    img = np.zeros((Y, X), np.uint8)
    # coarse checker
    img[Y // 2:, :X // 2] = 1
    img[:Y // 2, X // 2:] = 1
    # fine checker
    img[coarse_checker_size:-coarse_checker_size,
        coarse_checker_size:-coarse_checker_size] = 0
    img[coarse_checker_size:-coarse_checker_size:2,
        coarse_checker_size:-coarse_checker_size:2] = 1
    img[1 + coarse_checker_size:-coarse_checker_size:2,
        1 + coarse_checker_size:-coarse_checker_size:2] = 1
    # target
    img[total_checker_size:-total_checker_size,
        total_checker_size:-total_checker_size] = target_center
    if repeat > 1:
        img = img.repeat(repeat, axis=0).repeat(repeat, axis=1)
    return img * 255


def save_target(out, target):
    np.savetxt(out, target, delimiter=",", fmt="%d")
