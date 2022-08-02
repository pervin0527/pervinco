import os
import tensorflow as tf
from model import CenterNet
from data_loader import DataGenerator

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    epochs = 500
    batch_size = 32
    classes = ["face"]
    max_detections = 10
    backbone = "resnet18"
    learning_rate = 0.0001
    input_shape = (512, 512, 3)
    
    train_txt = "./data/custom_train.txt"
    test_txt = "./data/custom_test.txt"

    train_dataset = DataGenerator(train_txt, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    train_steps = int(tf.math.ceil(len(train_dataset) / batch_size).numpy())

    test_dataset = DataGenerator(test_txt, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    test_steps = int(tf.math.ceil(len(test_dataset) / batch_size).numpy())

    model = CenterNet(inputs=input_shape, num_classes=len(classes), backbone=backbone)
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=200,
                                                            t_mul=2.0,
                                                            m_mul=0.8,
                                                            alpha=learning_rate * 0.1)

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=test_dataset,
              validation_steps=test_steps,
              epochs = epochs)