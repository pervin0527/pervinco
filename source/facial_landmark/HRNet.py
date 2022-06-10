import tensorflow as tf

def hrnet_stem(filters=64):
    stem_layers = [tf.keras.layers.Conv2D(filters, 3, 2, 'same'),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.Conv2D(filters, 3, 2, 'same'),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.Activation('relu')]

    def forward(x):
        for layer in stem_layers:
            x = layer(x)
        return x

    return forward


def bottleneck_block(filters=64, expantion=1, kernel_size=(3, 3), downsample=False, padding='same', activation='relu'):
    filters *= expantion
    strides = (2, 2) if downsample else (1, 1)
    block_layers = [tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(1, 1),
                                           strides=strides,
                                           padding=padding,
                                           activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(activation)]

    block_layers.extend([tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=(1, 1),
                                                padding=padding,
                                                activation=None),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.Activation(activation)])

    block_layers.extend([tf.keras.layers.Conv2D(filters=filters * 4,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding=padding,
                                                activation=None),
                         tf.keras.layers.BatchNormalization()])

    matching_layers = [tf.keras.layers.Conv2D(filters=filters * 4,
                                              kernel_size=(1, 1),
                                              strides=strides,
                                              padding=padding,
                                              activation=None),
                       tf.keras.layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        # Match the feature map size and channels.
        for layer in matching_layers:
            inputs = layer(inputs)

        x = tf.keras.layers.Add()([x, inputs])

        # Finally, output of the block.
        x = tf.keras.layers.Activation(activation)(x)

        return x

    return forward


def hrn_1st_stage(filters=64, activation='relu'):
    block_layers = [bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding='same'),
                    tf.keras.layers.BatchNormalization()]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def fusion_layer(filters, upsample=False, activation='relu'):
    block_layers = []
    if upsample:
        block_layers.extend([tf.keras.layers.Conv2D(filters=filters,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding='same'),
                             tf.keras.layers.UpSampling2D(size=(2, 2),
                                                 interpolation='bilinear')])
    else:
        block_layers.append(tf.keras.layers.Conv2D(filters=filters,
                                                   kernel_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='same'))

    block_layers.extend([tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.Activation(activation)])

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def fusion_block(filters, branches_in, branches_out, activation='relu'):
    _fusion_grid = []

    rows = branches_in
    columns = branches_out

    for row in range(rows):
        _fusion_layers = []
        for column in range(columns):
            if column == row:
                _fusion_layers.append(tf.identity)
            elif column > row:
                # Down sampling.
                _fusion_layers.append(fusion_layer(filters * pow(2, column),
                                                   False, activation))
            else:
                # Up sampling.
                _fusion_layers.append(fusion_layer(filters * pow(2, column),
                                                   True, activation))

        _fusion_grid.append(_fusion_layers)

    if len(_fusion_grid) > 1:
        _add_layers_group = [tf.keras.layers.Add() for _ in range(branches_out)]

    def forward(inputs):
        rows = len(_fusion_grid)
        columns = len(_fusion_grid[0])

        fusion_values = [[None for _ in range(columns)] for _ in range(rows)]

        for row in range(rows):
            for column in range(columns):
                if column < row:
                    continue
                elif column == row:
                    x = inputs[row]
                elif column > row:
                    x = fusion_values[row][column - 1]

                fusion_values[row][column] = _fusion_grid[row][column](x)

            for column in reversed(range(columns)):
                if column >= row:
                    continue

                x = fusion_values[row][column + 1]
                fusion_values[row][column] = _fusion_grid[row][column](x)

        if rows == 1:
            outputs = [fusion_values[0][0], fusion_values[0][1]]
        else:
            outputs = []
            fusion_values = [list(v) for v in zip(*fusion_values)]

            for index, values in enumerate(fusion_values):
                outputs.append(_add_layers_group[index](values))

        return outputs

    return forward


def residual_block(filters=64, downsample=False, kernel_size=(3, 3), padding='same', activation='relu'):
    strides = (2, 2) if downsample else (1, 1)
    block_layers = [tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(activation)]

    block_layers.extend([tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=(1, 1),
                                                padding=padding,
                                                activation=None),
                         tf.keras.layers.BatchNormalization()])

    if downsample:
        matching_layers = [tf.keras.layers.Conv2D(filters=filters,
                                                  kernel_size=(1, 1),
                                                  strides=strides,
                                                  padding=padding,
                                                  activation=None),
                           tf.keras.layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        if downsample:
            for layer in matching_layers:
                inputs = layer(inputs)

        x = tf.keras.layers.Add()([x, inputs])
        x = tf.keras.layers.Activation(activation)(x)

        return x

    return forward


def hrn_block(filters=64, activation='relu'):
    block_layers = [residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation)]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def hrn_blocks(repeat=1, filters=64, activation='relu'):
    block_layers = [hrn_block(filters, activation=activation)
                    for _ in range(repeat)]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def hrnet_body(filters=64):
    # Stage 1
    s1_b1 = hrn_1st_stage(filters)
    s1_fusion = fusion_block(filters, branches_in=1, branches_out=2)

    # Stage 2
    s2_b1 = hrn_block(filters)
    s2_b2 = hrn_block(filters*2)
    s2_fusion = fusion_block(filters, branches_in=2, branches_out=3)

    # Stage 3
    s3_b1 = hrn_blocks(4, filters)
    s3_b2 = hrn_blocks(4, filters*2)
    s3_b3 = hrn_blocks(4, filters*4)
    s3_fusion = fusion_block(filters, branches_in=3, branches_out=4)

    # Stage 4
    s4_b1 = hrn_blocks(3, filters)
    s4_b2 = hrn_blocks(3, filters*2)
    s4_b3 = hrn_blocks(3, filters*4)
    s4_b4 = hrn_blocks(3, filters*8)

    def forward(inputs):
        # Stage 1
        x = s1_b1(inputs)
        x = s1_fusion([x])

        # Stage 2
        x_1 = s2_b1(x[0])
        x_2 = s2_b2(x[1])
        x = s2_fusion([x_1, x_2])

        # Stage 3
        x_1 = s3_b1(x[0])
        x_2 = s3_b2(x[1])
        x_3 = s3_b3(x[2])
        x = s3_fusion([x_1, x_2, x_3])

        # Stage 4
        x_1 = s4_b1(x[0])
        x_2 = s4_b2(x[1])
        x_3 = s4_b3(x[2])
        x_4 = s4_b4(x[3])

        return [x_1, x_2, x_3, x_4]

    return forward


def hrnet_heads(input_channels=64, output_channels=17):
    # Construct up sacling layers.
    scales = [2, 4, 8]
    up_scale_layers = [tf.keras.layers.UpSampling2D((s, s)) for s in scales]
    concatenate_layer = tf.keras.layers.Concatenate(axis=3)
    heads_layers = [tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')]

    def forward(inputs):
        scaled = [f(x) for f, x in zip(up_scale_layers, inputs[1:])]
        x = concatenate_layer([inputs[0], scaled[0], scaled[1], scaled[2]])
        for layer in heads_layers:
            x = layer(x)
        return x

    return forward


def hrnet(input_shape, output_channels, width=18):
    last_stage_width = sum([width * pow(2, n) for n in range(4)])

    inputs = tf.keras.Input(input_shape, dtype=tf.float32)
    x = hrnet_stem(64)(inputs)
    x = hrnet_body(width)(x)
    outputs = hrnet_heads(input_channels=last_stage_width, output_channels=output_channels)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model