import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

IMG_HEIGHT = 227
IMG_WIDTH = 227
CHANNEL = 3

def load_image(test_path, target_size):
    img = tf.keras.preprocessing.image.load_img(test_path, target_size=target_size)
    img_tensor=tf.keras.preprocessing.image.img_to_array(img)

    return img_tensor[np.newaxis]/255


def show_first_feature_map(loaded_model, img_path):
    first_output = loaded_model.layers[1].output
    # print(first_output.shape, first_output.dtype)

    model = tf.keras.models.Model(inputs=loaded_model.input, outputs=first_output)
    target_size = (loaded_model.input.shape[1], loaded_model.input.shape[2])
    img_tensor = load_image(img_path, target_size)

    # print(loaded_model.input.shape)
    # print(img_tensor)

    first_activation = model.predict(img_tensor)

    # print(first_activation.shape)  # (1, 148, 148, 32)
    # print(first_activation[0, 0, 0])  # [0.00675746 0. 0.02397328 0.03818807 0. ...]

    plt.figure(figsize=(16, 8))
    for i in range(first_activation.shape[-1]):
        plt.subplot(8, 12, i + 1)
        plt.axis('off')
        plt.matshow(first_activation[0, :, :, i], cmap='gray', fignum=0)
    plt.tight_layout()
    plt.show()


def predict_and_get_outputs(model, img_path):
    layer_outputs = [layer.output for layer in model.layers[1:11]]
    layer_names = [layer.name for layer in model.layers[1:11]]

    print([str(output.shape) for output in layer_outputs])
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    input_shape = (model.input.shape[1], model.input.shape[2])      # (150, 150)
    img_tensor = load_image(img_path, target_size=input_shape)

    for layer in layer_outputs:
        print(layer.shape)
    print('-' * 50)

    layer_outputs = activation_model.predict(img_tensor)
    for layer in layer_outputs:
        print(layer.shape)

    return layer_outputs, layer_names


def show_activation_maps(layer, title, layer_index, n_cols=16):
    size, n_features = layer.shape[1], layer.shape[-1]
    assert n_features % n_cols == 0

    n_rows = n_features // n_cols

    big_image = np.zeros((n_rows*size, n_cols*size), dtype=np.float32)

    for row in range(n_rows):
        for col in range(n_cols):
            channel = layer[0, :, :, row * n_cols + col]      # shape : (size, size)

            channel -= channel.mean()
            channel /= channel.std()
            channel *= 64
            channel += 128
            channel = np.clip(channel, 0, 255).astype('uint8')

            big_image[row*size:(row+1)*size, col*size:(col+1)*size] = channel

    plt.figure(figsize=(n_cols, n_rows))

    plt.xticks(np.arange(n_cols) * size)
    plt.yticks(np.arange(n_rows) * size)
    plt.title('layer {} : {}'.format(layer_index, title))
    plt.tight_layout()
    plt.imshow(big_image)           # cmap='gray'


if __name__ == '__main__':
    model_path = '/home/barcelona/pervinco/model/' \
                 'good/max99_2class_2019.12.20_16:03:04/ALEX1_2class_2019.12.20_16:03:04.h5'
    model = tf.keras.models.load_model(model_path)
    model.summary()

    test_path = '/home/barcelona/pervinco/datasets/predict/cat_dog/dog/google_0002.jpg'
    # show_first_feature_map(model, test_path)

    layer_outputs, layer_names = predict_and_get_outputs(model, test_path)

    for i, (layer, name) in enumerate(zip(layer_outputs, layer_names)):
        show_activation_maps(layer, name, i)

    plt.show()
