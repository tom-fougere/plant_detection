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

list_files = os.listdir(IMAGE_FOLDER)


def test_plot_masks():

    scale_percent = 0.6

    plot_masks(MASK_FOLDER, nb_masks=3, scale_percent=scale_percent)


def test_concatenate_image_with_mask():
    example_image = cv2.imread(IMAGE_FOLDER + list_files[IMG_ID])
    example_mask = cv2.imread(MASK_FOLDER + list_files[IMG_ID])

    merge = concatenate_image_with_mask(example_image, example_mask)

    assert merge.shape == (example_image.shape[0]*3, example_image.shape[1], example_image.shape[2])


def test_concatenate_several_images_with_mask():

    nb_images = 5
    example_images = []
    example_masks = []

    for i_images in range(nb_images):
        example_images.append(cv2.imread(IMAGE_FOLDER + list_files[i_images]))
        example_masks.append(cv2.imread(MASK_FOLDER + list_files[i_images]))

    merge = concatenate_several_images_with_mask(example_images, example_masks)

    assert merge.shape == (example_images[0].shape[0]*3,
                           example_images[0].shape[1]*nb_images,
                           example_images[0].shape[2])


def test_plot_images_with_masks():

    scale_percent = 0.6
    nb_images = 5
    example_images = []
    example_masks = []

    for i_images in range(nb_images):
        example_images.append(cv2.imread(IMAGE_FOLDER + list_files[i_images]))
        example_masks.append(cv2.imread(MASK_FOLDER + list_files[i_images]))

    plot_images_with_masks(example_images, example_masks, scale_percent)


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
