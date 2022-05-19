import tensorflow as tf
from glob import glob
from hparams_config import send_params

params = send_params(show_contents=False)
img_size = params["IMG_SIZE"]
num_classes = len(params["CLASSES"])
one_hot = params["ONE_HOT"]

def get_file_list(path):
    images = sorted(glob(f"{path}/images/*.jpg"))
    masks = sorted(glob(f"{path}/masks/*.png"))
    
    n_images, n_masks = len(images), len(masks)
    
    return images, masks, n_images, n_masks


def read_image(image_path, num_classes, img_size, one_hot_encoding, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(images=image, size=[img_size, img_size])
        image.set_shape([img_size, img_size, 1])

        if one_hot_encoding:
            image = tf.cast(image, tf.uint8)
            image = tf.squeeze(image, axis=-1)
            image = tf.one_hot(image, num_classes)

    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(images=image, size=[img_size, img_size])
        image.set_shape([img_size, img_size, 3])
        image = image / 127.5 - 1

    return image


def load_data(image_list, mask_list):
    image = read_image(image_list, num_classes, img_size, one_hot, mask=False)
    mask = read_image(mask_list, num_classes, img_size, one_hot, mask=True)

    return image, mask


def data_generator(image_list, mask_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset