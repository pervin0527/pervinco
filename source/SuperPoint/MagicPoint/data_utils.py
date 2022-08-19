import tensorflow as tf
import tensorflow_addons as tfa
import photometric_augmentation as photaug
from homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points

def add_dummy_valid_mask(data):
    with tf.name_scope('dummy_valid_mask'):
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


def photometric_augmentation(data, config):
    primitives = config["data"]["augmentation"]["photometric"]["primitives"]
    params = config["data"]["augmentation"]["photometric"]["params"]

    prim_configs = [params.get(p, {}) for p in primitives]
    indices = tf.range(len(primitives))
    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c)) for j, (p, c) in enumerate(zip(primitives, prim_configs))]
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


def flat2mat(H):
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def invert_homography(H):
    return mat2flat(tf.linalg.inv(flat2mat(H)))


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