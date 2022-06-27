import tensorflow as tf

def model_openpose_a2a_v2(INPUT_SIZE = 200, LANDMARK_SIZE = 55):
    #------------------------- feature map
    x = tf.keras.Input(shape=(None, None, 3), name='input_layer')
    h = tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'same')(x)
    h = tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.MaxPool2D()(h)

    h = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.MaxPool2D()(h)

    h = tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.MaxPool2D()(h)

    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)

    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same')(h)
    feature_map = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding = 'same', name='feature_map')(h)

    #------------------------- stage1
    h = tf.keras.layers.Conv2D(512, kernel_size=(1,1), activation='relu', name='stage_1')(feature_map)
    s1 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s1')(h)

    #------------------------- stage2
    h = tf.keras.layers.concatenate([feature_map, h], axis=-1, name='stage_2')
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu')(h)
    s2 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s2')(h)

    #------------------------- stage3
    h = tf.keras.layers.concatenate([feature_map, h], axis=-1, name='stage_3')
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu')(h)
    s3 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s3')(h)

    #------------------------- stage4
    h = tf.keras.layers.concatenate([feature_map, h], axis=-1, name='stage_4')
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu')(h)
    s4 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s4')(h)

    #------------------------- stage5
    h = tf.keras.layers.concatenate([feature_map, h], axis=-1, name='stage_5')
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu')(h)
    s5 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s5')(h)

    #------------------------- stage6
    h = tf.keras.layers.concatenate([feature_map, h], axis=-1, name='stage_6')
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(7,7), activation='relu', padding = 'same')(h)
    h = tf.keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu')(h)
    s6 = tf.keras.layers.Conv2D(LANDMARK_SIZE, kernel_size=(1,1), name='s6')(h)

    #------------------------- return
    output = h = tf.keras.layers.concatenate([s1, s2, s3, s4, s5, s6], axis=-1, name='output_layer')

    return tf.keras.Model(inputs=[x], outputs=[output])