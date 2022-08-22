import cv2
from math import pi
import tensorflow as tf
import tensorflow_addons as tfa

def sample_homography(shape,
                      perspective=True, 
                      scaling=True, 
                      rotation=True, 
                      translation=True,
                      n_scales=5, 
                      n_angles=25, 
                      scaling_amplitude=0.1, 
                      perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, 
                      patch_ratio=0.5, 
                      max_angle=pi/2,
                      allow_artifacts=False,
                      translation_overflow=0.):

    margin = (1 -patch_ratio) / 2
    pts1 = margin + tf.constant([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], tf.float32)
    pts2 = pts1

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = tf.random.truncated_normal([1], 0., perspective_amplitude_y/2)
        h_displacement_left = tf.random.truncated_normal([1], 0., perspective_amplitude_x/2)
        h_displacement_right = tf.random.truncated_normal([1], 0., perspective_amplitude_x/2)
        pts2 += tf.stack([tf.concat([h_displacement_left, perspective_displacement], 0),
                          tf.concat([h_displacement_left, -perspective_displacement], 0),
                          tf.concat([h_displacement_right, perspective_displacement], 0),
                          tf.concat([h_displacement_right, -perspective_displacement], 0)])

    if scaling:
        scales = tf.concat([[1.], tf.random.truncated_normal([n_scales], 1, scaling_amplitude/2)], 0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(tf.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = tf.range(1, n_scales + 1)
        else:
            valid = tf.where(tf.reduce_all((scaled >= 0.) & (scaled < 1.), [1, 2]))[:, 0]
        idx = valid[tf.random.uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += tf.expand_dims(tf.stack([tf.random.uniform((), -t_min[0], t_max[0]),
                                         tf.random.uniform((), -t_min[1], t_max[1])]), axis=0)

    if rotation:
        angles = tf.linspace(tf.constant(-max_angle), tf.constant(max_angle), n_angles)
        angles = tf.concat([[0.], angles], axis=0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles),
                                       tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(tf.tile(tf.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = tf.range(1, n_angles + 1)
        else:
            valid = tf.where(tf.reduce_all((rotated >= 0.) & (rotated < 1.),
                                           axis=[1, 2]))[:, 0]
        idx = valid[tf.random.uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = rotated[idx]

    shape = tf.cast(shape[::-1], tf.float32)
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    def ax(p, q): 
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): 
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(tf.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = tf.transpose(tf.linalg.lstsq(a_mat, p_mat, fast=True))
    
    return homography


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    mask = tfa.image.transform(tf.ones(image_shape), homography, interpolation="nearest")

    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,)*2)
        mask = tf.nn.erosion2d(value=mask[tf.newaxis, ..., tf.newaxis],
                               filters=tf.cast(tf.constant(kernel)[..., tf.newaxis], tf.float32),
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               data_format="NHWC",
                               dilations=[1, 1, 1, 1])[0, ..., 0] + 1.

    return tf.cast(mask, tf.int32)


def filter_points(points, shape):
    with tf.name_scope('filter_points'):
        mask = (points >= 0) & (points <= tf.cast(shape-1, tf.float32))
        return tf.boolean_mask(points, tf.reduce_all(mask, -1))


def flat2mat(H):
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def invert_homography(H):
    return mat2flat(tf.linalg.inv(flat2mat(H)))


def warp_points(points, homography):
    H = tf.expand_dims(homography, axis=0) if len(homography.shape) == 1 else homography

    num_points = tf.shape(points)[0]
    points = tf.cast(points, tf.float32)[:, ::-1]
    points = tf.concat([points, tf.ones([num_points, 1], dtype=tf.float32)], -1)

    H_inv = tf.transpose(flat2mat(invert_homography(H)))
    warped_points = tf.tensordot(points, H_inv, [[1], [0]])
    warped_points = warped_points[:, :2, :] / warped_points[:, 2:, :]
    warped_points = tf.transpose(warped_points, [2, 0, 1])[:, :, ::-1]

    return warped_points[0] if len(homography.shape) == 1 else warped_points