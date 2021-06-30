import numpy as np

# 카메라 칼리브레이션 함수들

def cam2pixel(cam_coord, f, c, scaling_factor):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None] * scaling_factor, y[:,None] * scaling_factor, z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c, scaling_factor):
    x = (pixel_coord[:, 0] - c[0] * scaling_factor) / (f[0] * scaling_factor) * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1] * scaling_factor) / (f[1] * scaling_factor) * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord

def pred2pixel(pred_coord, orig_img_shape, cropped_img_shape, pred_shape, factor_for_norm_depth):
    pred_coord[:, 0] = pred_coord[:, 0] / pred_shape * cropped_img_shape
    pred_coord[:, 1] = pred_coord[:, 1] / pred_shape * cropped_img_shape
    pred_coord[:, 0] += (orig_img_shape[0] - cropped_img_shape) / 2
    pred_coord[:, 1] += (orig_img_shape[1] - cropped_img_shape) / 2
    pred_coord[:, 2] /= pred_shape
    pred_coord[:, 2] = pred_coord[:, 2] * 2. - 1.
    pred_coord[:, 2] *= factor_for_norm_depth
    return pred_coord


