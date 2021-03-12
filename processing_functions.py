import tensorflow as tf
import numpy as np
import cv2


# Pre-processing of images and masks
def preprocessing_masks(mask):
    img = mask > 0
    img = tf.cast(img, tf.float32)
    return img


def convert_1channel_to_3channels(image):
    return np.concatenate((image,)*3, axis=-1)


def convert_rgb_mask_to_1channel_mask(image):
    img = image[:, :, 0:1]
    return img


def convert_rgb_mask_for_displaying(rgb_mask, max_value=1.):
    img = (rgb_mask * (255./max_value)).astype(np.uint8)
    return img
