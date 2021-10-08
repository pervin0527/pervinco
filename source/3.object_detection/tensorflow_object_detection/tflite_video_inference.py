import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# GPU setup
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
    model_file_path = "/data/Models/efficientdet_lite/efdet_dmc_d0_set1.tflite"
    video_file_path = "/data/Datasets/Seeds/DMC/samples/sample_video_1.mp4"
    label_file_path = "/data/Datasets/Seeds/DMC/labels/labels.txt"
    threshold = 0.7

    LABEL_FILE = pd.read_csv(label_file_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()

    interpreter = tf.lite.Interpreter(model_path=model_file_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0].get('shape')
    input_width, input_height = input_shape[1], input_shape[2]
    input_dtype = input_details[0].get('dtype')

    cap = cv2.VideoCapture(video_file_path)
    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv2.resize(frame, (input_width, input_height))
        input_tensor = np.expand_dims(frame, 0)
        
        interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.uint8))
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])

        boxes = boxes[0]
        classes = classes[0]
        scores = scores[0]

        for index, score in enumerate(scores):
            if score > threshold:
                box = boxes[index]
                label = CLASSES[int(classes[index])]

                ymin, xmin, ymax, xmax = int(box[0] * input_width), int(box[1] * input_width), int(box[2] * input_width), int(box[3] * input_width)

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))
                cv2.putText(frame, f"{label} {score:.2f}%", (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

        cv2.imshow('result', frame)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            os.system('clear')
            break

cap.release()
cv2.destroyAllWindows()