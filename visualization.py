import numpy as np
import cv2
import os

from processing_functions import postprocessing_masks_rgb


def plot_masks(mask_folder_path, nb_masks=3, scale_percent=0.6):

    list_masks = os.listdir(mask_folder_path)

    for i in range(nb_masks):
        current_mask = cv2.imread(mask_folder_path + list_masks[i])
        current_mask = current_mask * 100
        layers = np.concatenate((current_mask[:, :, 0], current_mask[:, :, 1], current_mask[:, :, 2]), axis=1)

        # New size
        width = int(layers.shape[1] * scale_percent)
        height = int(layers.shape[0] * scale_percent)
        new_dim = (width, height)

        # Rescale
        layers = cv2.resize(layers, new_dim)

        if i > 0:
            images = np.concatenate((images, layers), axis=0)
        else:
            images = layers

    # Plot image
    cv2.imshow('Several Masks', images)
    cv2.waitKey()


# Function to plot an image with its mask
def plot_image_with_mask(image, mask):
    mask_norm = postprocessing_masks_rgb(mask, image.shape)

    mixed_image = cv2.addWeighted(image, 0.5, mask_norm, 0.5, 0)
    raw_images = np.concatenate((image, mask_norm), axis=1)
    example = np.concatenate((raw_images, mixed_image), axis=1)

    # Plot image
    cv2.imshow('Left:Image, center:Mask, right:Superposition', example)
    cv2.waitKey()
