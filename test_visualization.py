import tensorflow as tf

from visualization import *
from ae_models import fcn8
from processing_functions import preprocessing_images
from manage_dataset import get_data

IMG_ID = 1
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/rgb/'
MASK_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/gt/'
HEIGHT = 256
WIDTH = 256


def test_plot_masks():

    plot_masks(MASK_FOLDER)


def test_plot_image_with_mask():

    list_files = os.listdir(IMAGE_FOLDER)

    example_image = cv2.imread(IMAGE_FOLDER + list_files[IMG_ID])
    example_mask = cv2.imread(MASK_FOLDER + list_files[IMG_ID])

    plot_image_with_mask(example_image, example_mask)


def test_plot_prediction():

    model = fcn8(HEIGHT, WIDTH, 3, 1, n_filters=32)
    model.load_weights('fcn8_weights.h5')
    example_image, example_mask = get_data('dataset/synthetic_sugarbeet_random_weeds/train_test/test/images/img/',
                                           'dataset/synthetic_sugarbeet_random_weeds/train_test/test/masks/img/',
                                           id=0)

    # Preprocessing image to get the right dim
    image_processed = preprocessing_images(example_image, new_size=(HEIGHT, WIDTH))
    image_tensor = tf.expand_dims(image_processed, axis=0)

    # Predict output with this image
    prediction = model.predict(image_tensor)
    pred_image = np.array(tf.squeeze(prediction, axis=0))

    plot_prediction(example_image, pred_image, example_mask)
