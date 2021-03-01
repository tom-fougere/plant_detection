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


def test_FCN8():

    test_convs, test_img_input = FCN8(input_height=HEIGHT, input_width=WIDTH, input_layer=LAYER)
    test_model = tf.keras.Model(inputs=test_img_input, outputs=[test_convs, test_img_input])

    # Check number of layers
    # 1 Input, 6*5 conv_block, 1 zeroPadding
    assert len(test_model.layers) == 32

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check first conv layer
    assert test_model.layers[2].output_shape == (None, HEIGHT, WIDTH, 32)

    # Check output
    assert test_model.layers[-1].output_shape == (None, HEIGHT/32, WIDTH/32, 256)

    # Check number of parameters
    assert test_model.count_params() == 2355360

    del test_convs, test_img_input, test_model


def test_dfcn8():

    kernel_size = 4
    stride_size = 2

    # start the encoder using the default input size
    test_convs, test_img_input = FCN8(input_height=HEIGHT, input_width=WIDTH, input_layer=LAYER)

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
    test_output = unet_encoder(test_input)
    test_model = tf.keras.Model(inputs=test_input, outputs=test_output)

    print(test_model.summary())

    # Check number of layers
    # 1 Input and 4 blocks of 6 layers (2 Conv2d, 2 activation, 1 Max pooling, 1 dropout)
    assert len(test_model.layers) == 6 * 4 + 1

    # Check input
    assert test_model.layers[0].output_shape[0] == (None, HEIGHT, WIDTH, LAYER)

    # Check layers size
    assert test_model.layers[1].output_shape == (None, HEIGHT, WIDTH, 64)
    assert test_model.layers[7].output_shape == (None, HEIGHT/2, WIDTH/2, 128)
    assert test_model.layers[13].output_shape == (None, HEIGHT/4, WIDTH/4, 256)
    assert test_model.layers[19].output_shape == (None, HEIGHT/8, WIDTH/8, 512)

    # Check layers type
    assert isinstance(test_model.layers[5], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[6], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[11], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[12], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[17], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[18], tf.keras.layers.Dropout)
    assert isinstance(test_model.layers[23], tf.keras.layers.MaxPool2D)
    assert isinstance(test_model.layers[24], tf.keras.layers.Dropout)

    # Check number of parameters
    assert test_model.count_params() == 4685376

    # free up test resources
    del test_input, test_output, test_model