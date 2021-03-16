import tensorflow as tf
from cv2 import imread
from visualization import *
from ae_models import fcn8
from processing_functions import convert_rgb_mask_to_1channel_mask
from manage_dataset import get_data

IMG_ID = 1
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/rgb/'
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/train_test/test/images/img/'
MASK_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/gt/'
MASK_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/train_test/test/masks/img/'
HEIGHT = 256
WIDTH = 256

list_files = os.listdir(IMAGE_FOLDER)


def test_merge_images():
    image1 = imread(IMAGE_FOLDER + list_files[0])
    image2 = imread(IMAGE_FOLDER + list_files[1])
    image3 = imread(IMAGE_FOLDER + list_files[2])

    horizontal_image = merge_images([image1, image2, image3], axis=1)
    vertical_image = merge_images([image1, image2, image3], axis=0)

    assert horizontal_image.shape == (image1.shape[0], image1.shape[1]*3, image1.shape[2])
    assert vertical_image.shape == (image1.shape[0]*3, image1.shape[1], image1.shape[2])


def test_overlay_image():
    image1 = imread(IMAGE_FOLDER + list_files[0])
    image2 = imread(IMAGE_FOLDER + list_files[1])

    merged_image1 = overlay_image(image1, image2, ratio=0.5)
    merged_image2 = overlay_image(image1, image2, ratio=0.2)

    assert merged_image1.shape == (image1.shape[0], image1.shape[1], image1.shape[2])
    assert merged_image1[0, 0, 0] == (0.5 * image1[0, 0, 0] + 0.5 * image2[0, 0, 0]).astype("uint8")
    assert merged_image2[10, 10, 0] == (0.2 * image1[10, 10, 0] + 0.8 * image2[10, 10, 0]).astype("uint8")


def test_merge_image_with_mask():
    image = imread(IMAGE_FOLDER + list_files[IMG_ID])
    mask = imread(MASK_FOLDER + list_files[IMG_ID])

    mask = convert_rgb_mask_to_1channel_mask(mask)

    vert_image1 = merge_image_with_mask(image, mask, third_overlay=True, axis=0)
    vert_image2 = merge_image_with_mask(image, mask, third_overlay=False, axis=0)
    hori_image1 = merge_image_with_mask(image, mask, third_overlay=True, axis=1)
    hori_image2 = merge_image_with_mask(image, mask, third_overlay=False, axis=1)

    assert vert_image1.shape == (image.shape[0]*3, image.shape[1], image.shape[2])
    assert vert_image2.shape == (image.shape[0]*2, image.shape[1], image.shape[2])
    assert hori_image1.shape == (image.shape[0], image.shape[1]*3, image.shape[2])
    assert hori_image2.shape == (image.shape[0], image.shape[1]*2, image.shape[2])


def test_plot_masks_from_folder():

    scale_percent = 0.5
    plot_masks_from_folder(MASK_FOLDER, nb_masks=3, scale_percent=scale_percent)


def test_plot_images_and_masks():

    scale_percent = 0.5
    nb_images = 5
    images = []
    masks = []

    for i_images in range(nb_images):
        images.append(imread(IMAGE_FOLDER + list_files[i_images]))

        current_mask = imread(MASK_FOLDER + list_files[i_images])
        current_mask = convert_rgb_mask_to_1channel_mask(current_mask)  # formatting
        masks.append(current_mask)

    plot_images_and_masks(images, masks, third_overlay=True, scale_percent=scale_percent)


def test_plot_predictions():

    scale_percent = 1
    nb_images = 5
    images_list = []

    model = fcn8(HEIGHT, WIDTH, 3, 1, n_filters=32)
    model.load_weights('fcn8_weights.h5')

    model_input_dim = model.layers[0].output_shape[0]

    for i_images in range(nb_images):
        cur_image = imread(IMAGE_FOLDER + list_files[i_images])
        cur_image = resize(cur_image, (model_input_dim[1], model_input_dim[2]))
        images_list.append(cur_image)

    # Convert list of images into 4d array (and format them for model)
    images = np.asarray(images_list)/255.

    # Predict
    predictions = model.predict(images)

    # Convert tensor into list of images
    predictions_list = [predictions[i] for i in range(predictions.shape[0])]

    # Plot prediction
    plot_predictions(images_list, predictions_list, scale_percent=scale_percent, third_overlay=True)
