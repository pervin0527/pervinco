import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 1 filter
image = tf.constant([[
    [[1], [2], [3]],
    [[4], [5], [6]],
    [[7], [8], [9]]
]], dtype=np.float32)

print("image shape", image.shape)
plt.imshow(image.numpy().reshape(3, 3), cmap='Greys')
plt.show()

weight = np.array([[[[1.]], [[1.]]],
                  [[[1.]], [[1.]]]])
print("weight shape", weight.shape)

# Padding = VALID
weight_init = tf.constant_initializer(weight)
conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', kernel_initializer=weight_init)(image)
print('conv2d shape', conv2d.shape)
print(conv2d.numpy().reshape(2, 2))

plt.imshow(conv2d.numpy().reshape(2, 2), cmap='gray')
plt.show()

# Padding = SAME
weight_init = tf.constant_initializer(weight)
conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=2, padding='SAME', kernel_initializer=weight_init)(image)
print('conv2d shape', conv2d.shape)
print(conv2d.numpy().reshape(3, 3))

plt.imshow(conv2d.numpy().reshape(3, 3), cmap='gray')
plt.show()

# 3 filters
weights = np.array([[[[1., 10., -1.]], [[1., 10., -1.]]],
                  [[[1., 10., -1.]], [[1., 10., -1.]]]])

print('weights shape', weights.shape)

weights_init = tf.constant_initializer(weights)
conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME', kernel_initializer=weights_init)(image)
print('conv2d shape', conv2d.shape)

feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3, 3))
    plt.subplot(1, 3, i+1)
    plt.imshow(feature_map.reshape(3, 3), cmap='gray')
plt.show()