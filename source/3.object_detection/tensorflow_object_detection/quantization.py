import cv2
import pathlib
import numpy as np
import tensorflow as tf

# # GPU setup
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 1:
#     try:
#         print("Activate Multi GPU")
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#     except RuntimeError as e:
#         print(e)

# else:
#     try:
#         print("Activate Sigle GPU")
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         strategy = tf.distribute.experimental.CentralStorageStrategy()
#     except RuntimeError as e:
#         print(e)


def read_testset(lists):
    images = []
    for image in lists:
        image = cv2.imread(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        images.append(image)

    images = np.array(images)
    print(images.shape)
    return images

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

if __name__ == "__main__":
    IMG_SIZE = 512
    testset_path = "/data/Datasets/testset/fire/"
    label_file_paths='/data/Datasets/Seeds/fire/labels.txt'
    path = "/home/barcelona/tensorflow/models/research/object_detection/custom/models/fire/21_07_10/"
    output_path = path.split('/')[:-1]
    output_path = '/'.join(output_path)
    saved_model_dir = f"{path}/saved_model"

    test_path = pathlib.Path(testset_path)
    test_path = list(test_path.glob('*.jpg'))
    test_image_list = sorted([str(path) for path in test_path])
    raw_test_data = read_testset(test_image_list)

    tflite_models_dir = pathlib.Path(f'{path}')
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.allow_custom_ops= True
    converter.experimental_new_converter= True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter.convert()

    with open(f'{output_path}/efficientdet_d0_512x512_integer_quant.tflite', 'wb') as w:
        w.write(tflite_quant_model)
        
    print("Integer Quantization complete! - efficientdet_d0_512x512_integer_quant.tflite")