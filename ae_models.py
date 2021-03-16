import tensorflow as tf
from tensorflow.keras.models import model_from_json

# parameter describing where the channel dimension is found in our dataset
IMAGE_ORDERING = 'channels_last'
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_LAYER = 3


def load_model(model_name, weights_name):
    """
    Load model with adapted weights
    :param model_name: path of the json model
    :param weights_name: path of the weights
    :return: model
    """
    # Load the model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights in the model
    model.load_weights(weights_name)
    print("Loaded model from disk")

    return model

############################
## FCN-8
############################


def conv_block(input, filters, strides=3, pooling_size=2, pool_strides=2):
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


def fcn8_encoder(input_height=DEFAULT_HEIGHT, input_width=DEFAULT_WIDTH, input_layer=DEFAULT_LAYER,
                 n_filters=32, strides=3, pooling_size=2, pool_strides=2):
    """
    Defines the down-sampling path of the image segmentation model.

    Args:
      input_height (int) -- height of the images
      input_width (int) -- width of the images
      input_layer (int) -- number of layers of the images
      n_filters (int) -- number of filters of the Conv2D layers
      strides (int) -- strides setting of the Conv2D layers
      pooling_size (int) -- pooling size of the MaxPooling2D layers
      pool_strides (int) -- strides setting of the MaxPooling2D layers

    Returns:
    (tuple of tensors, tensor)
      tuple of tensors -- features extracted at blocks 3 to 5
      tensor -- copy of the input
    """

    img_input = tf.keras.layers.Input(shape=(input_height, input_width, input_layer))

    # pad the input image to have dimensions to the nearest power of two
    x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(img_input)

    # Block 1
    x = conv_block(x, filters=n_filters, strides=strides, pooling_size=pooling_size, pool_strides=pool_strides)

    # Block 2
    x = conv_block(x, filters=n_filters*2, strides=strides, pooling_size=pooling_size, pool_strides=pool_strides)

    # Block 3
    x = conv_block(x, filters=n_filters*4, strides=strides, pooling_size=pooling_size, pool_strides=pool_strides)
    f3 = x

    # Block 4
    x = conv_block(x, filters=n_filters*8, strides=strides, pooling_size=pooling_size, pool_strides=pool_strides)
    f4 = x

    # Block 5
    x = conv_block(x, filters=n_filters*8, strides=strides, pooling_size=pooling_size, pool_strides=pool_strides)
    f5 = x

    return (f3, f4, f5), img_input


def fcn8_decoder(convs, n_classes, n_filters=512, dropout=0.3):
    """
    Defines the up-sampling path of the image segmentation model.

    Args:
    convs (tuple of tensors) -- features extracted at blocks 3 to 5
    n_classes (int) -- number of classes in the label map
    n_filters (int) -- number of filters of the Conv2D layers
    dropout (float) -- Fraction of the input units to drop

    Returns:
      outputs (tensor) -- the pixel wise label map of the image
    """

    # features from the encoder stage
    f3, f4, f5 = convs

    # add convolutional layers on top of the CNN extractor.
    o = tf.keras.layers.Conv2D(n_filters, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING)(f5)
    o = tf.keras.layers.Dropout(dropout)(o)

    o = tf.keras.layers.Conv2D(n_filters, kernel_size=1, activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = tf.keras.layers.Dropout(dropout)(o)

    o = tf.keras.layers.Conv2D(n_classes, kernel_size=1, activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)

    # Up-sample `o` above and crop any extra pixels introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=4, strides=2, use_bias=False)(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f4
    o2 = tf.keras.layers.Conv2D(n_classes, kernel_size=1, activation='relu', padding='same', data_format=IMAGE_ORDERING)(o2)

    # add the results of the up-sampling and pool 4 prediction
    o = tf.keras.layers.Add()([o, o2])

    # Up-sample the resulting tensor of the operation you just did
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=4, strides=2, use_bias=False)(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f3
    o2 = tf.keras.layers.Conv2D(n_classes, kernel_size=1, activation='relu', padding='same', data_format=IMAGE_ORDERING)(o2)

    # add the results of the upsampling and pool 3 prediction
    o = tf.keras.layers.Add()([o, o2])

    # Up-sample up to the size of the original image
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=8, strides=8, use_bias=False)(o)
    o = tf.keras.layers.Cropping2D(((0, 0), (0, 0)))(o)

    # append a sigmoid activation
    outputs = (tf.keras.layers.Activation('sigmoid'))(o)

    return outputs


def fcn8(input_height=DEFAULT_HEIGHT, input_width=DEFAULT_WIDTH, input_layer=DEFAULT_LAYER, output_layer=DEFAULT_LAYER,
         n_filters=32, pool_size=2, pool_strides=2, strides=3, dropout=0.5):
    """
    Defines the FCN-8 by connecting the encoder and decoder.

    Args:
    input_height (int) -- dimension of the height of the inputted image
    input_width (int) -- dimension of the width of the inputted image
    input_layer (int) -- number of layers of the inputted image
    output_layer (int) -- number of layers in the last layer (or number of classes in the label map)
    n_filters (int) -- number of filters
    pool_size (int) -- Windows size over which to take the maximum
    pool_strides (int) -- strides setting of the MaxPooling2D layers
    strides (int) -- strides for the deconvolution/upsampling
    dropout (float) -- Fraction of the input units to drop

    Returns:
      model (tensor) -- the model of the U-net network
    """

    # start the encoder using the default input size 64 x 84
    convs, img_input = fcn8_encoder(input_height, input_width, input_layer,
                                    n_filters=n_filters, strides=strides,
                                    pooling_size=pool_size, pool_strides=pool_strides)

    # pass the convolutions obtained in the encoder to the decoder
    dec_op = fcn8_decoder(convs, output_layer, n_filters=n_filters*6, dropout=dropout)

    # define the model specifying the input (batch of images) and output (decoder output)
    model = tf.keras.Model(inputs=img_input, outputs=dec_op)

    return model


############################
##  UNET
############################


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    """
    Adds 2 convolutional layers with the parameters passed to it

    Args:
      input_tensor (tensor) -- the input tensor
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size for the convolution

    Returns:
      tensor of output features
    """
    NB_CONV2D_LAYER = 2

    tensor = input_tensor
    for i_layer in range(NB_CONV2D_LAYER):
        tensor = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same',
                                        kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(tensor)
        tensor = tf.keras.layers.Activation('relu')(tensor)

    return tensor


def unet_encoder_block(input_tensor, n_filters=64, pool_size=2, dropout=0.3, kernel_size=3):
    """
    Adds two convolutional blocks and then perform down sampling on output of convolutions.

    Args:
    input_tensor (tensor) -- the input tensor
    n_filters (int) -- number of filters
    pool_size (int) -- Windows size over which to take the maximum
    dropout (float) -- Fraction of the input units to drop
    kernel_size (int) -- kernel size for the convolution

    Returns:
    features - the output features of the convolution block
    outputs - the maxpooled features with dropout
    """

    features = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(features)
    outputs = tf.keras.layers.Dropout(dropout)(p)

    return features, outputs


def unet_encoder(inputs, nb_filters=64, dropout=0.3, pool_size=2):
    """
    This function defines the encoder or downsampling path.

    Args:
    inputs (tensor) -- batch of input images
    n_filters (int) -- number of filters
    pool_size (int) -- Windows size over which to take the maximum
    dropout (float) -- Fraction of the input units to drop

    Returns:
    outputs - the output maxpooled features of the last encoder block
    (features1, features2, features3, features4) - the output features of all the encoder blocks
    """

    features1, p1 = unet_encoder_block(inputs, n_filters=nb_filters, pool_size=pool_size, dropout=dropout)
    features2, p2 = unet_encoder_block(p1, n_filters=nb_filters*2, pool_size=pool_size, dropout=dropout)
    features3, p3 = unet_encoder_block(p2, n_filters=nb_filters*4, pool_size=pool_size, dropout=dropout)
    features4, p4 = unet_encoder_block(p3, n_filters=nb_filters*8, pool_size=pool_size, dropout=dropout)
    outputs = p4

    return outputs, (features1, features2, features3, features4)


def unet_bottleneck(inputs, nb_filters=1024):
    """
    This function defines the bottleneck convolutions to extract more features before the upsampling layers.

    Args:
    n_filters (int) -- number of filters
    """

    bottle_neck = conv2d_block(inputs, n_filters=nb_filters)

    return bottle_neck


def unet_decoder_block(inputs, features, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    """
    defines the one decoder block of the UNet

    Args:
    inputs (tensor) -- batch of input features
    features (tensor) -- features from an encoder block
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size
    strides (int) -- strides for the deconvolution/upsampling
    dropout (float) -- Fraction of the input units to drop

    Returns:
    outputs (tensor) -- output features of the decoder block
    """

    tensor = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size,
                                             strides=strides, padding='same')(inputs)
    tensor = tf.keras.layers.concatenate([tensor, features])
    tensor = tf.keras.layers.Dropout(dropout)(tensor)
    outputs = conv2d_block(tensor, n_filters=n_filters, kernel_size=kernel_size)

    return outputs


def unet_decoder(inputs, features, n_classes, n_filters=64, kernel_size=3, strides=2, dropout=0.3):
    """
    Defines the decoder of the UNet chaining together 4 decoder blocks.

    Args:
    inputs (tensor) -- batch of input features
    features (tuple) -- features from the encoder blocks
    n_classes (int) -- number of classes in the label map
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size
    strides (int) -- strides for the deconvolution/upsampling
    dropout (float) -- Fraction of the input units to drop

    Returns:
      outputs (tensor) -- the pixel wise label map of the image
    """

    f1, f2, f3, f4 = features

    c6 = unet_decoder_block(inputs, f4, n_filters=n_filters*8, kernel_size=kernel_size, strides=strides, dropout=dropout)
    c7 = unet_decoder_block(c6, f3, n_filters=n_filters*4, kernel_size=kernel_size, strides=strides, dropout=dropout)
    c8 = unet_decoder_block(c7, f2, n_filters=n_filters*2, kernel_size=kernel_size, strides=strides, dropout=dropout)
    c9 = unet_decoder_block(c8, f1, n_filters=n_filters, kernel_size=kernel_size, strides=strides, dropout=dropout)

    outputs = tf.keras.layers.Conv2D(n_classes, kernel_size=1, activation='softmax')(c9)

    return outputs


def unet(input_height=DEFAULT_HEIGHT, input_width=DEFAULT_WIDTH, input_layer=DEFAULT_LAYER, output_layer=DEFAULT_LAYER,
         n_filters=64, pool_size=2, kernel_size=3, strides=2, dropout=0.3):
    """
    Defines the UNet by connecting the encoder, bottleneck and decoder.

    Args:
    input_height (int) -- dimension of the height of the inputted image
    input_width (int) -- dimension of the width of the inputted image
    input_layer (int) -- number of layers of the inputted image
    output_layer (int) -- number of layers in the last layer (or number of classes in the label map)
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size
    pool_size (int) -- Windows size over which to take the maximum
    strides (int) -- strides for the deconvolution/upsampling
    dropout (float) -- Fraction of the input units to drop

    Returns:
      model (tensor) -- the model of the U-net network
    """

    # specify the input shape
    inputs = tf.keras.layers.Input(shape=(input_height, input_width, input_layer))

    # feed the inputs to the encoder
    encoder_output, features_conv = unet_encoder(inputs,
                                                 nb_filters=n_filters,
                                                 dropout=dropout,
                                                 pool_size=pool_size)

    # feed the encoder output to the bottleneck
    bottle_neck = unet_bottleneck(encoder_output, n_filters*16)

    # feed the bottleneck and encoder block outputs to the decoder
    outputs = unet_decoder(bottle_neck,
                           features_conv,
                           n_classes=output_layer,
                           n_filters=n_filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           dropout=dropout)

    # create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
