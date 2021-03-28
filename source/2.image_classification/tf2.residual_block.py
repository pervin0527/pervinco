import tensorflow as tf

inputs = tf.keras.Input(shape=(28, 28, 256))

conv1 = tf.keras.layers.Conv2D(filter=64, kernel_size=1, padding='SAME', activation=tf.keras.layers.ReLU())(inputs)
conv2 = tf.keras.layers.Conv2D(filter=64, kernel_size=3, padding='SAME', activation=tf.keras.layers.ReLU())(conv1)
conv3 = tf.keras.layers.Conv2D(filter=256, kernel_size=1, padding='SAME')(conv2)
add3 = tf.keras.layers.add([conv3, inputs])
relu3 = tf.keras.layers.ReLU()(add3)

model = tf.keras.Model(inputs=inputs, outputs=relu3)