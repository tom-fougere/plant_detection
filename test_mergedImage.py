import os
import numpy as np
from cv2 import imread
from my_image import *


IMG_ID = 1
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/rgb/'
IMAGE_FOLDER = 'dataset/synthetic_sugarbeet_random_weeds/train_test/test/images/img/'

list_files = os.listdir(IMAGE_FOLDER)


def test_add_one_image():
    image1 = imread(IMAGE_FOLDER + list_files[IMG_ID])

    vertical_image = MergedImage()
    vertical_image.add_image(image1, axis=0)
    horizontal_image = MergedImage()
    horizontal_image.add_image(image1, axis=1)

    assert vertical_image.nb_lines == 1
    assert vertical_image.nb_columns == 1
    assert horizontal_image.nb_lines == 1
    assert horizontal_image.nb_columns == 1

    np.testing.assert_equal(vertical_image.images['l0c0'], image1)
    np.testing.assert_equal(horizontal_image.images['l0c0'], image1)


def test_add_two_images():
    image1 = imread(IMAGE_FOLDER + list_files[IMG_ID])
    image2 = imread(IMAGE_FOLDER + list_files[IMG_ID+1])

    vertical_image = MergedImage()
    vertical_image.add_image(image1, axis=0)
    vertical_image.add_image(image2, axis=0)
    horizontal_image = MergedImage()
    horizontal_image.add_image(image1, axis=1)
    horizontal_image.add_image(image2, axis=1)

    assert vertical_image.nb_lines == 2
    assert vertical_image.nb_columns == 1
    assert horizontal_image.nb_lines == 1
    assert horizontal_image.nb_columns == 2

    np.testing.assert_equal(vertical_image.images['l0c0'], image1)
    np.testing.assert_equal(vertical_image.images['l1c0'], image2)
    np.testing.assert_equal(horizontal_image.images['l0c0'], image1)
    np.testing.assert_equal(horizontal_image.images['l0c1'], image2)


def test_add_several_images():
    image1 = imread(IMAGE_FOLDER + list_files[IMG_ID])
    image2 = imread(IMAGE_FOLDER + list_files[IMG_ID+1])
    image3= imread(IMAGE_FOLDER + list_files[IMG_ID+2])

    full_image = MergedImage()
    full_image.add_images([image1, image2, image3], axis=0)

    assert full_image.nb_lines == 3
    assert full_image.nb_columns == 1

    np.testing.assert_equal(full_image.images['l0c0'], image1)
    np.testing.assert_equal(full_image.images['l1c0'], image2)
    np.testing.assert_equal(full_image.images['l2c0'], image3)


def test_add_merged_images_line():
    image1 = imread(IMAGE_FOLDER + list_files[IMG_ID])
    image2 = imread(IMAGE_FOLDER + list_files[IMG_ID + 1])
    image3 = imread(IMAGE_FOLDER + list_files[IMG_ID + 2])
    image4 = imread(IMAGE_FOLDER + list_files[IMG_ID + 3])
    image5 = imread(IMAGE_FOLDER + list_files[IMG_ID + 4])

    first_merged_image = MergedImage()
    first_merged_image.add_images([image1, image2, image3], axis=0)

    second_merged_image = MergedImage()
    second_merged_image.add_images([image4, image5], axis=0)

    first_merged_image.add_merged_image(second_merged_image, axis=0)

    assert first_merged_image.nb_lines == 5
    assert first_merged_image.nb_columns == 1

    np.testing.assert_equal(first_merged_image.images['l3c0'], image4)
    np.testing.assert_equal(first_merged_image.images['l4c0'], image5)


def test_add_merged_images_column():
    image1 = imread(IMAGE_FOLDER + list_files[IMG_ID])
    image2 = imread(IMAGE_FOLDER + list_files[IMG_ID + 1])
    image3 = imread(IMAGE_FOLDER + list_files[IMG_ID + 2])
    image4 = imread(IMAGE_FOLDER + list_files[IMG_ID + 3])
    image5 = imread(IMAGE_FOLDER + list_files[IMG_ID + 4])

    first_merged_image = MergedImage()
    first_merged_image.add_images([image1, image2, image3], axis=0)

    second_merged_image = MergedImage()
    second_merged_image.add_images([image4, image5], axis=0)

    first_merged_image.add_merged_image(second_merged_image, axis=1)

    assert first_merged_image.nb_lines == 3
    assert first_merged_image.nb_columns == 2

    np.testing.assert_equal(first_merged_image.images['l0c1'], image4)
    np.testing.assert_equal(first_merged_image.images['l1c1'], image5)

