import os
import sys
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from model import CenterNet
from IPython.display import clear_output
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=sys.maxsize)
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.float32)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if len(gt_boxes) > 0:
        gt_boxes[:, [0,2]] = gt_boxes[:, [0,2]] * nw / iw + dw
        gt_boxes[:, [1,3]] = gt_boxes[:, [1,3]] * nh / ih + dh
        gt_boxes[:, 0:2][gt_boxes[:, 0:2]<0] = 0
        gt_boxes[:, 2][gt_boxes[:, 2]>w] = w
        gt_boxes[:, 3][gt_boxes[:, 3]>h] = h
        box_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        box_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_boxes = gt_boxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box

    return image_paded, gt_boxes


def process_data(line):
    line = line.decode("utf-8").split()
    image = np.array(cv2.imread(line[0]))
    labels = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    image, labels = image_preporcess(np.copy(image), [input_shape[0],input_shape[1]], np.copy(labels))

    output_h = input_shape[0] // 4
    output_w = input_shape[1] // 4
    hm = np.zeros((output_h, output_w, len(classes)),dtype=np.float32)
    wh = np.zeros((max_detections, 2),dtype=np.float32)
    reg = np.zeros((max_detections, 2),dtype=np.float32)
    ind = np.zeros((max_detections),dtype=np.float32)
    reg_mask = np.zeros((max_detections),dtype=np.float32)

    for idx, label in enumerate(labels):
        bbox = label[:4]/4
        class_id = label[4]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h),math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[:,:,class_id], ct_int, radius)
        wh[idx] = 1. * w, 1. * h
        ind[idx] = ct_int[1] * output_w + ct_int[0]
        reg[idx] = ct - ct_int
        reg_mask[idx] = 1
    
    return image, hm, wh, reg, reg_mask, ind


def get_data(batch_lines):
    batch_image = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.float32)
    batch_hm = np.zeros((batch_size, input_shape[0]//4, input_shape[1]//4, len(classes)), dtype=np.float32)
    batch_wh = np.zeros((batch_size, max_detections, 2), dtype=np.float32)
    batch_reg = np.zeros((batch_size, max_detections, 2), dtype=np.float32)
    batch_reg_mask = np.zeros((batch_size, max_detections), dtype=np.float32)
    batch_indices = np.zeros((batch_size, max_detections), dtype=np.float32)

    for num, line in enumerate(batch_lines):
        image, hm, wh, reg, reg_mask, indices = process_data(line)
        batch_image[num, :, :, :] = image
        batch_hm[num, :, :, :] = hm
        batch_wh[num, :, :] = wh
        batch_reg[num, :, :] = reg
        batch_reg_mask[num, :] = reg_mask
        batch_indices[num, :] = indices

    return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_indices


def load_dataset(txt_path):
    num_dataset = len(open(txt_path, 'r').readlines())
    steps_per_epoch = int(math.ceil(float(num_dataset)) / batch_size)

    dataset = tf.data.TextLineDataset(txt_path)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x : tf.numpy_function(get_data, inp=[x], 
                                    Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda image, hm, wh, reg, reg_mask, indices : {"image" : image, "hm" : hm, "wh" : wh, "reg" : reg, "reg_mask" : reg_mask, "indices" : indices})
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)                                    

    return dataset, steps_per_epoch


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


def plot_predictions(model):
    image = cv2.imread(f"{data_dir}/dog.jpg")
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    input_tensor = np.expand_dims(image, axis=0)
    hm_pred, wh_pred, reg_pred = model.predict(input_tensor)


if __name__ == "__main__":
    data_dir = "/home/ubuntu/Datasets/COCO2017"
    label_dir = f"{data_dir}/Labels/labels.txt"
    train_txt_dir = f"{data_dir}/train.txt"
    test_txt_dir = f"{data_dir}/test.txt"
    
    backbone = "resnet18"
    epochs = 1000
    batch_size = 32
    threshold = 0.1
    max_detections = 100
    input_shape = [512, 512, 3]

    df = pd.read_csv(label_dir, sep=",", index_col=False, header=None)
    classes = df[0].to_list()
    
    train_dataset, train_steps = load_dataset(train_txt_dir)
    test_dataset, test_steps = load_dataset(test_txt_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ModelCheckpoint("/home/ubuntu/Models/test.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    model = CenterNet(input_shape, len(classes), max_detections, threshold, backbone, False)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=test_dataset,
              validation_steps=test_steps,
              callbacks=callbacks,
              epochs=epochs)