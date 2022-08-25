import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import augmentation.photometric_augmentation as photoaug
from tensorflow_addons.image import transform as H_transform
from augmentation.homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points

def add_dummy_valid_mask(data):
    valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    image_shape = tf.shape(data['image'])[:2]
    kp = tf.minimum(tf.cast(tf.round(data['keypoints']), tf.int32), image_shape-1)
    kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    
    return {**data, 'keypoint_map': kmap}


def ratio_preserving_resize(image, config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.cast(tf.divide(target_size, tf.shape(image)[:2]), tf.float32)
    new_size = tf.cast(tf.shape(image)[:2], tf.float32) * tf.reduce_max(scales)
    image = tf.image.resize(image, tf.cast(new_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])


def downsample(image, coordinates, **config):
    k_size = config['blur_size']
    kernel = cv2.getGaussianKernel(k_size, 0)[:, 0]
    kernel = np.outer(kernel, kernel).astype(np.float32)
    kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
    pad_size = int(k_size/2)
    image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
    image = tf.expand_dims(image, axis=0)  # add batch dim
    image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    ratio = tf.divide(tf.convert_to_tensor(config['resize']), tf.shape(image)[0:2])
    coordinates = coordinates * tf.cast(ratio, tf.float32)
    image = tf.image.resize(image, config['resize'], method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates


def flat2mat(H):
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def invert_homography(H):
    return mat2flat(tf.linalg.inv(flat2mat(H)))


def photometric_augmentation(data, config):
    primitives = config["data"]["augmentation"]["photometric"]["primitives"]
    params = config["data"]["augmentation"]["photometric"]["params"]

    prim_configs = [params.get(p, {}) for p in primitives]
    indices = tf.range(len(primitives))
    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photoaug, p)(image, **c)) for j, (p, c) in enumerate(zip(primitives, prim_configs))]
        image = tf.case(fn_pairs)
        return i + 1, image

    _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)), step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, config, add_homography=False):
    params = config["data"]["augmentation"]["homographic"]["params"]
    valid_border_margin = config["data"]["augmentation"]["homographic"]["valid_border_margin"]

    image_shape = tf.shape(data["image"])[:2]
    homography = sample_homography(image_shape, **params)[0]
    warped_image = tfa.image.transform(data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, valid_border_margin)

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points, 'valid_mask': valid_mask}
    if add_homography:
        ret["homography"] = homography
    return ret


def box_nms(prob, size, iou=0.1, threshold=0.01, keep_top_k=0):
    pts = tf.cast(tf.where(tf.greater_equal(prob, threshold)), dtype=tf.float32)
    size = tf.constant(size/2.)
    boxes = tf.concat([pts-size, pts+size], axis=1)
    scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))
    
    indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)
    pts = tf.gather(pts, indices)
    scores = tf.gather(scores, indices)
    if keep_top_k:
        k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
        scores, indices = tf.nn.top_k(scores, k)
        pts = tf.gather(pts, indices)
    prob = tf.scatter_nd(tf.cast(pts, tf.int32), scores, tf.shape(prob))
    
    return prob


def homography_adaptation(image, model, config):
    image = image[tf.newaxis, ...]
    probs = model(image)[1]
    counts = tf.ones_like(probs)
    images = image

    probs = tf.expand_dims(probs, axis=-1)
    counts = tf.expand_dims(counts, axis=-1)
    images = tf.expand_dims(images, axis=-1)

    shape = tf.shape(image)[1:3]

    def step(i, probs, counts, images):
        H = sample_homography(shape, config["model"]['homography_adaptation']['homographies'])
        H_inv = invert_homography(H)
        warped = H_transform(image, H, interpolation='BILINEAR')
        count = H_transform(tf.expand_dims(tf.ones(tf.shape(image)[:3]), -1), H_inv, interpolation='NEAREST')
        mask = H_transform(tf.expand_dims(tf.ones(tf.shape(image)[:3]), -1), H, interpolation='NEAREST')
        
        if config["model"]["homography_adaptation"]['valid_border_margin']:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["model"]["homography_adaptation"]['valid_border_margin'] * 2,) * 2)
            
            count = tf.nn.erosion2d(value=count,
                                    filters=tf.cast(tf.constant(kernel)[..., tf.newaxis], dtype=tf.float32),
                                    strides=[1, 1, 1, 1], 
                                    dilations=[1, 1, 1, 1],
                                    padding='SAME',
                                    data_format="NHWC")[..., 0] + 1.
            mask = tf.nn.erosion2d(value=mask,
                                   filters=tf.cast(tf.constant(kernel)[..., tf.newaxis], dtype=tf.float32),
                                   strides=[1, 1, 1, 1],
                                   dilations=[1, 1, 1, 1],
                                   padding='SAME',
                                   data_format="NHWC")[..., 0] + 1.

        prob = model(warped)[1]
        prob = prob * mask
        prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv, interpolation='BILINEAR')[..., 0]
        prob_proj = prob_proj * count

        probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
        counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
        images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
        return i + 1, probs, counts, images

    _, probs, counts, images = tf.nest.map_structure(tf.stop_gradient, 
                                                     tf.while_loop(lambda i, p, c, im : tf.less(i, config["model"]["homography_adaptation"]["num"] - 1), 
                                                                   step,
                                                                   [0, probs, counts, images],
                                                                   parallel_iterations=1,
                                                                   shape_invariants=[tf.TensorShape([]),
                                                                                   tf.TensorShape([None, None, None, None]),
                                                                                   tf.TensorShape([None, None, None, None]),
                                                                                   tf.TensorShape([None, None, None, 1, None])]))
    counts = tf.reduce_sum(counts, axis=-1)
    max_prob = tf.reduce_max(probs, axis=-1)
    mean_prob = tf.reduce_sum(probs, axis=-1) / counts

    if config["model"]["homography_adaptation"]['aggregation'] == 'max':
        prob = max_prob
    elif config["model"]["homography_adaptation"]['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['model']['homography_adaptation']['aggregation']))

    if config['model']['homography_adaptation']['filter_counts']:
        prob = tf.where(tf.greater_equal(counts, config['model']['homography_adaptation']['filter_counts']), prob, tf.zeros_like(prob))

    return {'prob': prob, 'counts': counts, 'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug