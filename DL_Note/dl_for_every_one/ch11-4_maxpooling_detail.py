import numpy as np
import tensorflow as tf

image = tf.constant([[[[4], [3]],
                     [[2], [1]]]], dtype=np.float32)

# padding = 'VALID'
pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='VALID')(image)

print(pool.shape)
print(pool.numpy())

# padding = 'SAME'

pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='SAME')(image)

print(pool.shape)
print(pool.numpy())