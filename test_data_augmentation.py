import cv2
import numpy as np

from data_augmentation import *
from processing_functions import preprocessing_masks

target_folder = "dataset/synthetic_sugarbeet_random_weeds/train_test/"
batch_size = 10
target_size = (256, 256)
mask_preprocessing_function = preprocessing_masks

train_image_generator, train_mask_generator = create_train_generator(target_folder,
                                                                     batch_size,
                                                                     target_size,
                                                                     mask_preprocessing_function)


def test_generators():

    my_train_generator = my_image_mask_generator(train_image_generator, train_mask_generator)

    # Display one couple image/mask from generators
    for (img, mask, trash) in my_train_generator:
        # Image
        cur_image = img[0] * 255
        cur_image = cur_image.astype(np.uint8)
        cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)

        # Mask
        cur_mask = np.zeros(shape=cur_image.shape)
        cur_mask[:, :, 0] = cur_mask[:, :, 1] = cur_mask[:, :, 2] = mask[0][:, :, 0] * 255.

        # Create concatenation
        image_mask = np.concatenate((cur_image, cur_mask), axis=1)
        image_mask = image_mask.astype(np.uint8)

        # Plot image
        cv2.imshow('Example from generator', image_mask)
        cv2.waitKey(0)

        break
