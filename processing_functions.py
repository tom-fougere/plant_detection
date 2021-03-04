import tensorflow as tf
import numpy as np
import cv2


# Pre-processing of images and masks
def preprocessing_masks(mask):
    img = mask > 0
    img = tf.cast(img, tf.float32)
    return img


def preprocessing_images(image, new_size):
    img = image.astype(np.float32)
    img = cv2.resize(image, (new_size[1], new_size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    return img


# Post processing of images and masks for visualization
def postprocessing_masks_rgb(mask, new_size):
    img = mask>0
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (new_size[1], new_size[0]))
    return img


def postprocessing_masks_prediction(mask, new_size):
    img = np.zeros(shape=(mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = mask[:, :, 0]
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (new_size[1], new_size[0]))
    return img


def postprocessing_images(image, new_size):
    img = cv2.resize(image, (new_size[1], new_size[0]))
    return img
