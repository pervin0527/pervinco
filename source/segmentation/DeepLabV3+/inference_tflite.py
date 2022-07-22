import cv2
import numpy as np
import tensorflow as tf

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


def decode_segmentation_masks(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, len(COLORMAP)):
        idx = mask == l
        r[idx] = COLORMAP[l, 0]
        g[idx] = COLORMAP[l, 1]
        b[idx] = COLORMAP[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)

    return overlay


if __name__ == "__main__":
    model_path = "/data/Models/segmentation/VOC2012-ResNet101-AUGMENT_50/saved_model/VOC2012-ResNet101-AUGMENT_50.tflite"
    image_path = "./images/sample/airplane.jpg"

    COLORMAP = [[0, 0, 0], # background
                [128, 0, 0], # aeroplane
                [0, 128, 0], # bicycle
                [128, 128, 0], # bird
                [0, 0, 128], # boat
                [128, 0, 128], # bottle
                [0, 128, 128], # bus
                [128, 128, 128], # car
                [64, 0, 0], # cat
                [192, 0, 0], # chair
                [64, 128, 0], # cow
                [192, 128, 0], # diningtable
                [64, 0, 128], # dog
                [192, 0, 128], # horse
                [64, 128, 128], # motorbike
                [192, 128, 128], # person
                [0, 64, 0], # potted plant
                [128, 64, 0], # sheep
                [0, 192, 0], # sofa
                [128, 192, 0], # train
                [0, 64, 128] # tv/monitor
    ]
    COLORMAP = np.array(COLORMAP, dtype=np.uint8)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

    image = cv2.imread(image_path)
    image = cv2.resize(image, (height, width))
    image = image / 127.5 - 1
    input_tensor = np.expand_dims(image, axis=0)

    # interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.uint8))
    interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])

    # whole_matrix = predictions[0]
    # for channel in range(len(COLORMAP)):
    #     matrix = whole_matrix[:,:,channel]
    #     print(matrix.shape)
    #     np.savetxt(f'./images/matrix/{channel}.txt', matrix, delimiter=',')

    if predictions[0].shape[-1] == len(COLORMAP):        
        predictions = np.argmax(predictions[0], axis=-1)
        decode_pred = decode_segmentation_masks(predictions)

    else:
        decode_pred = decode_segmentation_masks(predictions)

    overlay_image = get_overlay(image, decode_pred)

    cv2.imwrite("./images/sample/prediction.jpg", decode_pred)
    cv2.imwrite("./images/sample/overlay.jpg", overlay_image)
    cv2.imshow("result", overlay_image)
    cv2.waitKey(0)