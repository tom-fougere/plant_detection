import tensorflow as tf

# parameter describing where the channel dimension is found in our dataset
IMAGE_ORDERING = 'channels_last'
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_LAYER = 3


def conv_block(input, filters, strides, pooling_size, pool_strides):
    """
    Args:
      input (tensor) -- batch of images or features
      filters (int) -- number of filters of the Conv2D layers
      strides (int) -- strides setting of the Conv2D layers
      pooling_size (int) -- pooling size of the MaxPooling2D layers
      pool_strides (int) -- strides setting of the MaxPooling2D layers

    Returns:
      (tensor) max pooled and batch-normalized features of the input
    """

    # use the functional syntax to stack the layers as shown in the diagram above
    x = tf.keras.layers.Conv2D(filters, strides, padding='same', data_format=IMAGE_ORDERING)(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters, strides, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=pooling_size, strides=pool_strides)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def FCN8(input_height=DEFAULT_HEIGHT, input_width=DEFAULT_WIDTH, input_layer=DEFAULT_LAYER):
    """
    Defines the down-sampling path of the image segmentation model.

    Args:
      input_height (int) -- height of the images
      input_width (int) -- width of the images
      input_layer (int) -- number of layers of the images

    Returns:
    (tuple of tensors, tensor)
      tuple of tensors -- features extracted at blocks 3 to 5
      tensor -- copy of the input
    """

    img_input = tf.keras.layers.Input(shape=(input_height, input_width, input_layer))

    # pad the input image to have dimensions to the nearest power of two
    x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(img_input)

    # Block 1
    x = conv_block(x, filters=32, strides=(3, 3), pooling_size=(2, 2), pool_strides=(2, 2))

    # Block 2
    x = conv_block(x, filters=64, strides=(3, 3), pooling_size=(2, 2), pool_strides=(2, 2))

    # Block 3
    x = conv_block(x, filters=128, strides=(3, 3), pooling_size=(2, 2), pool_strides=(2, 2))
    f3 = x

    # Block 4
    x = conv_block(x, filters=256, strides=(3, 3), pooling_size=(2, 2), pool_strides=(2, 2))
    f4 = x

    # Block 5
    x = conv_block(x, filters=256, strides=(3, 3), pooling_size=(2, 2), pool_strides=(2, 2))
    f5 = x

    return (f3, f4, f5), img_input


def fcn8_decoder(convs, n_classes):
    # features from the encoder stage
    f3, f4, f5 = convs

    # number of filters
    n = 512

    # add convolutional layers on top of the CNN extractor.
    o = tf.keras.layers.Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING)(f5)
    o = tf.keras.layers.Dropout(0.5)(o)

    o = tf.keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING)(o)
    o = tf.keras.layers.Dropout(0.5)(o)

    o = tf.keras.layers.Conv2D(n_classes,  (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)

    # Up-sample `o` above and crop any extra pixels introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f4
    o2 = tf.keras.layers.Conv2D(n_classes, kernel_size=(1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o2)

    # add the results of the up-sampling and pool 4 prediction
    o = tf.keras.layers.Add()([o, o2])

    # Up-sample the resulting tensor of the operation you just did
    o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(4, 4),  strides=(2, 2), use_bias=False )(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f3
    o2 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o2)

    # add the results of the upsampling and pool 3 prediction
    o = tf.keras.layers.Add()([o, o2])

    # Up-sample up to the size of the original image
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(8, 8),  strides=(8, 8), use_bias=False)(o)
    o = tf.keras.layers.Cropping2D(((0, 0), (0, 0)))(o)

    # append a sigmoid activation
    o = (tf.keras.layers.Activation('sigmoid'))(o)

    return o
