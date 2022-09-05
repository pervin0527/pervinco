import os
import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

if __name__ == "__main__":
    img_path = "/data/Datasets/SPC/Testset/Normal/images/0002.jpg"
    image = cv2.imread(img_path)
    image = cv2.resize(image, (384, 384))
    result_img = image.copy()
    input_tensor = np.expand_dims(image, axis=0)

    pb_path = "/data/Models/efficientdet_lite/full-name13-GAP6-300/saved_model"
    model = tf.saved_model.load(pb_path)

    output = model(input_tensor)
    # print(output)

    bboxes = output[0].numpy()
    scores = output[1].numpy()
    classes = output[2].numpy()
    print(bboxes.shape)
    print(scores.shape)
    print(classes.shape)

    for idx, score in enumerate(scores[0]):
        if score > 0.7:
            bbox = bboxes[0][idx]
            label = classes[0][idx]

            cv2.rectangle(result_img, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), color=(0, 0, 255), thickness=3)

    cv2.imshow("result", result_img)
    cv2.waitKey(0)        