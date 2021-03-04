import os
from visualization import *

IMG_ID = 1
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/rgb/'
MASK_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/gt/'


def test_plot_masks():

    plot_masks(MASK_FOLDER)


def test_plot_image_with_mask():

    list_files = os.listdir(IMAGE_FOLDER)

    example_image = cv2.imread(IMAGE_FOLDER + list_files[IMG_ID])
    example_mask = cv2.imread(MASK_FOLDER + list_files[IMG_ID])

    plot_image_with_mask(example_image, example_mask)
