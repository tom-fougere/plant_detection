import numpy as np
from cv2 import imread, resize, imshow, waitKey
import os
import random
import matplotlib.pyplot as plt

from processing_functions import convert_1channel_to_3channels, convert_rgb_mask_for_displaying


def merge_images(images: list, axis=0):
    """
    Merge list of images
    :param images: List of images, np-array
    :param axis: Axis to merge (0 = vertical, 1 = horizontal), integer
    :return: merge of all images in the list, np-array
    """
    return np.concatenate(images, axis=axis).astype(np.uint8)


def overlay_image(image1, image2, ratio=0.5):
    """
    Overlay Two images
    :param image1: First image
    :param image2: Second image (must have the same size)
    :param ratio: The factor of blending the first image
    :return: Merged image
    """

    return (image1 * ratio + image2 * (1 - ratio)).astype("uint8")


def merge_image_with_mask(image, mask, third_overlay=False, max_value_mask=1., axis=0):
    """
    Build an image with the real images and its mask
    It's possible to add (merge) an overlay of the images
    :param image: 3-channel image (X, X, 3), np-array
    :param mask: 1-channel image (X, X, 1), np-array
    :param third_overlay: Boolean to add the overlay image, bool
    :param max_value_mask: Maximal value in the mask, float
    :param axis: Axis to build the merge image including mask, integer
    :return: Merged image
    """

    mask_to_display = convert_1channel_to_3channels(mask)
    mask_to_display = convert_rgb_mask_for_displaying(mask_to_display, max_value=max_value_mask)

    # Merge the image with its mask
    merged_image = merge_images([image, mask_to_display], axis=axis)

    # Add third image is third_overlay is True
    if third_overlay:
        overlaying_image = overlay_image(image, mask_to_display)
        merged_image = merge_images([merged_image, overlaying_image], axis=axis)

    return merged_image


def plot_masks_from_folder(mask_folder_path, nb_masks=3, scale_percent=0.6):
    """
    Plot several masks (chosen randomly)
    :param mask_folder_path: Directory containing masks, string
    :param nb_masks: Number of masks to display, integer
    :param scale_percent: Scale of image to display, float, [0-1]
    :return:
    """

    list_masks = os.listdir(mask_folder_path)
    nb_files = len(list_masks)
    count = 0

    for i in random.sample(range(nb_files), nb_masks):
        count += 1
        current_mask = imread(mask_folder_path + list_masks[i])

        # Format
        current_mask = convert_rgb_mask_for_displaying(current_mask, max_value=1.)

        # New size
        width = int(current_mask.shape[1] * scale_percent)
        height = int(current_mask.shape[0] * scale_percent)
        new_dim = (width, height)

        # Rescale
        current_mask = resize(current_mask, new_dim)

        if count > 1:
            images = merge_images([images, current_mask], axis=1)
        else:
            images = current_mask

    # Plot images
    imshow('Binary masks', images)
    waitKey()


def plot_images_and_masks(images: list, masks: list, third_overlay=False, scale_percent=0.6):
    """
    Plot image(s) and mask(s)
    :param images: List of images, np-array
    :param masks: List of images, np-array
    :param third_overlay: Boolean to add the overlay image, True/False
    :param scale_percent: Scale of image to display, float, [0-1]
    :return:
    """

    full_image = merge_image_with_mask(images[0], masks[0], third_overlay=third_overlay, axis=0)

    for image, mask in zip(images[1:], masks[1:]):
        current_image = merge_image_with_mask(image, mask, third_overlay=third_overlay, axis=0)
        full_image = merge_images([full_image, current_image], axis=1)

    # New Dimension
    height = int(full_image.shape[0] * scale_percent)
    width = int(full_image.shape[1] * scale_percent)
    new_dim = (width, height)

    full_image = resize(full_image, new_dim)

    # Plot images
    imshow('Top:Image, Middle:Mask, Bottom:Overlay', full_image)
    waitKey()


def plot_predictions(images: list, predictions: list, third_overlay=False, scale_percent=0.6):
    """
    Plot predictions with inputted image
    :param images: List of images (input of the NN), np-array
    :param predictions: List of masks (output of the NN), np-array
    :param third_overlay: Boolean to add the overlay image, True/False
    :param scale_percent: Scale of image to display, float, [0-1]
    :return:
    """

    full_image = merge_image_with_mask(images[0], predictions[0], third_overlay=third_overlay, axis=0)

    for image, pred in zip(images[1:], predictions[1:]):
        image_and_prediction = merge_image_with_mask(image, pred, third_overlay=third_overlay, axis=0)
        full_image = merge_images([full_image, image_and_prediction], axis=1)

    # New Dimension
    height = int(full_image.shape[0] * scale_percent)
    width = int(full_image.shape[1] * scale_percent)
    new_dim = (width, height)

    full_image = resize(full_image, new_dim)

    imshow('Predictions', full_image)
    waitKey()


def plot_loss_accuracy(model_history):
    """
    Plot loss and accuracy (for training epoch)
    :param model_history: History of training
    :return:
    """

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.legend(["Training", "Validation"])
    plt.title('Accuracy')

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.legend(["Training", "Validation"])
    plt.title('Loss')
