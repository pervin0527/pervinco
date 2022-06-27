import cv2
import numpy as np
import tensorflow as tf


def make_confidence_map(label, sigma = 2.5):
    norm = [MAP_SIZE/2, MAP_SIZE/2]
    new_label = label * norm + norm
    
    grid_x = np.tile(np.arange(MAP_SIZE), (MAP_SIZE, 1))
    grid_y = np.tile(np.arange(MAP_SIZE), (MAP_SIZE, 1)).transpose()
    grid_x = np.tile(np.expand_dims(grid_x, axis=-1), LANDMARK_SIZE)
    grid_y = np.tile(np.expand_dims(grid_y, axis=-1), LANDMARK_SIZE)
    
    grid_distance = (grid_x - new_label[:,0]) ** 2 + (grid_y - new_label[:,1]) ** 2
    confidence_map = np.exp(-1 * grid_distance / sigma ** 2) # why 0.5?
    
    return confidence_map.astype(np.float32)


def data_process(data):
    splits = tf.strings.split(data, sep=' ')
    image_path = splits[0]
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])

    label = splits[1:206]
    label = tf.strings.to_number(label, out_type=tf.float32)

    return image, label


if __name__ == "__main__":
    IMG_SIZE = 112
    MAP_SIZE = 100
    MAP_SIGMA = 2.5
    LANDMARK_SIZE = 98
    N_LANDMARKS = LANDMARK_SIZE * 2

    data_dir = "/data/Datasets/WFLW/train_data"
    image_dir = f"{data_dir}/imgs"
    annotation_dir = f"{data_dir}/list.txt"

    dataset = tf.data.TextLineDataset(annotation_dir)
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(make_confidence_map)

    for data in dataset.take(1):
        print(data)