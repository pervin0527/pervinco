import math
import itertools
import numpy as np
import tensorflow as tf

"""
math.sqrt : 제곱근
itertools.product('ABCD', repeat=2) -> AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
"""
def generate_default_boxes(config):
    default_boxes = []
    ratios = config['ratios'] # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    scales = config['scales'] # [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
    fm_sizes = config['fm_sizes'] # [38, 19, 10, 5, 3, 1]

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            # print(fm_size, i, j)
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size

            default_boxes.append([cx, cy, scales[m], scales[m]])
            default_boxes.append([cx, cy, math.sqrt(scales[m] * scales[m + 1]), math.sqrt(scales[m] * scales[m + 1])])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)

                default_boxes.append([cx, cy, scales[m] * r, scales[m] / r])
                default_boxes.append([cx, cy, scales[m] / r, scales[m] * r])

        # print(np.array(default_boxes).shape)

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 0.1)

    return default_boxes