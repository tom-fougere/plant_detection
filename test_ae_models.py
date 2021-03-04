from ae_models import *

HEIGHT = 64
WIDTH = 32
LAYER = 3


def test_conv_block():
    test_input = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, LAYER))
    test_output = conv_block(test_input, 32, 3, 2, 2)
    test_model = tf.keras.Model(inputs=test_input, outputs=test_output)

    # Check number of layers
    # 1 Input and 2 Conv2d, 2 LeakyRelu, 1 batch norm
    assert len(test_model.layers) == 7

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check first conv layer
    assert test_model.layers[1].output_shape == (None, HEIGHT, WIDTH, 32)

    # Check output
    assert test_model.layers[-1].output_shape == (None, HEIGHT/2, WIDTH/2, 32)

    # Check number of parameters
    assert test_model.count_params() == 10272

    # free up test resources
    del test_input, test_output, test_model


def test_encoder_fcn8():

    test_convs, test_img_input = fcn8_encoder(input_height=HEIGHT, input_width=WIDTH, input_layer=LAYER)
    test_model = tf.keras.Model(inputs=test_img_input, outputs=[test_convs, test_img_input])

    # Check number of layers
    # 1 Input, 6*5 conv_block, 1 zeroPadding
    assert len(test_model.layers) == 32

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check first conv layer
    assert test_model.layers[2].output_shape == (None, HEIGHT, WIDTH, 32)

    # Check output
    assert test_model.layers[-1].output_shape == (None, int(HEIGHT/32), int(WIDTH/32), 256)

    # Check number of parameters
    assert test_model.count_params() == 2355360

    del test_convs, test_img_input, test_model


def test_decodeur_fcn8():

    kernel_size = 4
    stride_size = 2

    # start the encoder using the default input size
    test_convs, test_img_input = fcn8_encoder(input_height=HEIGHT, input_width=WIDTH, input_layer=LAYER)

    # pass the convolutions obtained in the encoder to the decoder
    n_classes = 2
    test_fcn8_decoder = fcn8_decoder(test_convs, n_classes)

    # define the model specifying the input (batch of images) and output (decoder output)
    test_model = tf.keras.Model(inputs=test_img_input, outputs=test_fcn8_decoder)

    # Get dimensions of the last layer of the encoder
    dim_encoder = test_model.layers[31].output_shape

    # Check number of layers
    # encoder = 32, decoder = 11 = 3 conv2dTranspose, 3 cropping2d, 3 conv2d, 2 Add
    assert len(test_model.layers) == 32 + 11

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check first conv layer
    assert test_model.layers[2].output_shape == (None, HEIGHT, WIDTH, 32)

    # Check first layer of decoder
    assert test_model.layers[32].output_shape == (None, stride_size * (dim_encoder[1]-1) + kernel_size,
                                                  stride_size * (dim_encoder[2]-1) + kernel_size, 2)

    # Check output
    assert test_model.layers[-1].output_shape == (None, HEIGHT, WIDTH, 2)

    # Check number of parameters
    assert test_model.count_params() == 2364644

    del test_convs, test_img_input, test_fcn8_decoder, test_model


def test_conv2d_block():
    test_input = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, LAYER))
    test_output = conv2d_block(test_input, 32, 2)
    test_model = tf.keras.Model(inputs=test_input, outputs=test_output)

    # Check number of layers
    # 1 Input and 2 Conv2d, 2 activation
    assert len(test_model.layers) == 5

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check layers size
    assert test_model.layers[1].output_shape == (None, HEIGHT, WIDTH, 32)
    assert test_model.layers[3].output_shape == (None, HEIGHT, WIDTH, 32)

    # Check layers type
    assert isinstance(test_model.layers[1], tf.keras.layers.Conv2D)
    assert isinstance(test_model.layers[2], tf.keras.layers.Activation)
    assert isinstance(test_model.layers[3], tf.keras.layers.Conv2D)

    # Check number of parameters
    assert test_model.count_params() == 4544

    # free up test resources
    del test_input, test_output, test_model


def test_unet_encoder():
    test_input = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, LAYER))
    test_output, conv_features = unet_encoder(test_input)
    test_model = tf.keras.Model(inputs=test_input, outputs=test_output)

    # Check number of layers
    # 1 Input and 4 blocks of 6 layers (2 Conv2d, 2 activation, 1 Max pooling, 1 dropout)
    assert len(test_model.layers) == 6 * 4 + 1

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check layers size
    assert test_model.layers[1].output_shape == (None, HEIGHT, WIDTH, 64)
    assert test_model.layers[7].output_shape == (None, int(HEIGHT/2), int(WIDTH/2), 128)
    assert test_model.layers[13].output_shape == (None, int(HEIGHT/4), int(WIDTH/4), 256)
    assert test_model.layers[19].output_shape == (None, int(HEIGHT/8), int(WIDTH/8), 512)

    # Check layers type
    assert isinstance(test_model.layers[5], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[6], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[11], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[12], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[17], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[18], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[23], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[24], tf.keras.layers.Dropout)

    # Check conv features
    assert len(conv_features) == 4
    assert conv_features[0].shape.as_list() == [None, HEIGHT, WIDTH, 64]
    assert conv_features[1].shape.as_list() == [None, HEIGHT/2, WIDTH/2, 128]
    assert conv_features[2].shape.as_list() == [None, HEIGHT/4, WIDTH/4, 256]
    assert conv_features[3].shape.as_list() == [None, HEIGHT/8, WIDTH/8, 512]

    # Check number of parameters
    assert test_model.count_params() == 4685376

    # free up test resources
    del test_input, test_output, test_model


def test_unet_bottleneck():
    test_input = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, LAYER))
    test_output = unet_bottleneck(test_input)
    test_model = tf.keras.Model(inputs=test_input, outputs=test_output)

    # Check number of layers
    # 1 Input, 2 conv2d, 2 Activation
    assert len(test_model.layers) == 5

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check layers size
    assert test_model.layers[1].output_shape == (None, HEIGHT, WIDTH, 1024)

    # Check number of parameters
    assert test_model.count_params() == 9466880

    # free up test resources
    del test_input, test_output, test_model


def test_unet():
    test_model = unet(input_height=HEIGHT, input_width=WIDTH, input_layer=3, output_layer=2)

    count_conv2d = count_activation = count_maxpool = count_dropout = 0
    for i_layer in test_model.layers:
        if isinstance(i_layer, tf.keras.layers.Conv2D):
            count_conv2d += 1
        if isinstance(i_layer, tf.keras.layers.Activation):
            count_activation += 1
        if isinstance(i_layer, tf.keras.layers.MaxPool2D):
            count_maxpool += 1
        if isinstance(i_layer, tf.keras.layers.Dropout):
            count_dropout += 1

    # Check number of layers per type
    assert count_conv2d == 19 + 4  # 19 conv2D and 4 conv2d_transpose
    assert count_activation == 18
    assert count_maxpool == 4
    assert count_dropout == 4 * 2

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check layers size
    assert test_model.layers[-1].output_shape == (None, HEIGHT, WIDTH, 2)

    # free up test resources
    del test_model
