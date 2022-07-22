import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import img_to_array

def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
    original_img = np.asarray(image, dtype = np.float32)
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    with tf.GradientTape() as tape:
        gradient_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    activation_map = cv2.resize(activation_map.numpy(), 
                                (original_img.shape[1], 
                                 original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)


    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    #superimpose heatmap onto image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)

    plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    plt.show()
    # cv2.imshow('result', cvt_heatmap)
    # cv2.waitKey(0)

    # #enlarge plot
    # plt.rcParams["figure.dpi"] = 100

    # if plot_results == True:
    #     plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    # else:
    #     return cvt_heatmap

model = tf.keras.models.load_model("/data/Models/classification/SPC/2022.04.11_06:50")
testset = "/data/Datasets/SPC/Testset/Normal/images"
testset = sorted(glob(f"{testset}/*"))

for path in testset:
    image = cv2.imread(path)
    image = cv2.resize(image, (320, 320))
    VizGradCAM(model, img_to_array(image), plot_results=True)
