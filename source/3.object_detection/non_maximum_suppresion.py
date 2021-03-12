import cv2, os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from model import YOLOv1

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("ActivateMulti GPU")
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

def reshape_yolo_preds(preds):
  # 7x7x(20+5*2) = 1470 -> 7x7x30
  return tf.reshape(preds, [tf.shape(preds)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])

def convert(x, y, w, h):
    w = input_width * w
    h = input_height * h
    x = x * input_width
    y = y * input_height

    xmin = int(x - (w / 2))
    ymin = int(y - (h / 2))
    xmax = int(x + (w / 2))
    ymax = int(y + (h / 2))

    return xmin, ymin, xmax, ymax

def get_boxes_and_scores(prediction):
    boxes = np.empty((0, 4), float)
    scores = np.empty((0, 20), float)

    for i in range(cell_size):
        for j in range(cell_size):
            # print(prediction[i][i])

            for k in range(boxes_per_cell):
                x = prediction[i][j][0 + (5 * k)]
                y = prediction[i][j][1 + (5 * k)]
                w = prediction[i][j][2 + (5 * k)]
                h = prediction[i][j][3 + (5 * k)]
                confidence = prediction[i][j][4 + (5 * k)]
                softmax_reg = prediction[i][j][10:]

                # xmin, ymin, xmax, ymax = convert(x, y, w, h)
                # box = np.array([xmin, ymin, xmax, ymax])
                box = np.array([x, y, w, h])
                score = np.dot(confidence, softmax_reg)

                # print(box)
                # print(score)

                boxes = np.append(boxes, [box], axis=0)
                scores = np.append(scores, [score], axis=0)

        #     break
        # break
    return boxes, scores

def iou(boxA, boxB):
    # print(boxA, boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_maximum_suppression(boxes, scores, threshold, iou_threshold):
    # print(scores)
    for label in range(len(classes)):
        score = scores[:, label]

        tmp = []
        for b, s in zip(boxes, score):
            # print(b, s)
            tmp.append([b, s])

        # print(tmp[0])
        # print(tmp[0][1])

        for i in range(len(tmp)):
            if tmp[i][1] < threshold:
                tmp[i][1] = 0

        tmp.sort(key=lambda x : x[1], reverse=True)
        # print(tmp)

        for i in range(len(tmp)):
            if tmp[i] != 0:
                bbox_max = tmp[i]

                for j in range(i+1, len(tmp)):
                    bbox_cur = tmp[j]

                    if iou(bbox_max[0], bbox_cur[0]) < iou_threshold:
                        tmp[j][1] = 0

        # print(tmp)
        
        for i in range(len(tmp)):
            scores[i][label] = tmp[i][1]

        # print(scores[:, label])

    for label in range(len(classes)):
        print(scores[:, label])
        
if __name__ == "__main__":
    input_width = 224
    input_height = 224
    cell_size = 7
    num_classes = 20
    boxes_per_cell = 2
    threshold = 0.5
    iou_threshold = 0.5
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    image = '/data/backup/pervinco/test_code/dog.jpg'
    latest_ckpt = tf.train.latest_checkpoint('/data/backup/pervinco/test_code/ckpt-7750.index')
    YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))

    test_image = cv2.imread(image)
    test_image = cv2.resize(test_image, (224, 224))
    test_image = tf.expand_dims(test_image, axis=0)

    os.system('clear')
    predict = YOLOv1_model(test_image)
    predict = reshape_yolo_preds(predict)

    # 7 * 7 * 30
    prediction = predict[0].numpy()
    print(prediction[0][0])
    print(sum(prediction[0][0]) / 30)
    # print(prediction.shape)
    # # print(prediction)

    # boxes, scores = get_boxes_and_scores(prediction)
    # print(boxes.shape, scores.shape)

    # non_maximum_suppression(boxes, scores, threshold, iou_threshold)