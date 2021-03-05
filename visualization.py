import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from processing_functions import postprocessing_masks_rgb, postprocessing_masks_prediction, postprocessing_images


def plot_masks(mask_folder_path, nb_masks=3, scale_percent=0.6):
    """
    Display layers of several masks

    Args:
      mask_folder_path (string) - ground truth label maps
      nb_masks (int) - Number of masks to display
      scale_percent (float) - Scale of image to display
    """

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


def plot_image_with_mask(image, mask):
    """
    Display an image with its mask (and both)

    Args:
      image (cv2.Mat) - Image
      mask (cv2.Mat) - Mask
    """
    mask_norm = postprocessing_masks_rgb(mask, image.shape)

    mixed_image = cv2.addWeighted(image, 0.5, mask_norm, 0.5, 0)
    raw_images = np.concatenate((image, mask_norm), axis=1)
    example = np.concatenate((raw_images, mixed_image), axis=1)

    # Plot image
    cv2.imshow('Left:Image, center:Mask, right:Superposition', example)
    cv2.waitKey()


def plot_prediction(image, prediction, gt):
    """
    Display an image, the expected output and the prediction

    Args:
      image (cv2.Mat) - Image
      prediction (cv2.Mat) - Predicted output (mask)
      gt (cv2.Mat) - Expected output (mask)
    """

    image_example_gt = np.concatenate((postprocessing_images(image,
                                                             new_size=(prediction.shape[0],
                                                                       prediction.shape[1])),
                                       postprocessing_masks_rgb(gt,
                                                                new_size=(prediction.shape[0],
                                                                          prediction.shape[1])),
                                       postprocessing_masks_prediction(prediction,
                                                                       new_size=(prediction.shape[0],
                                                                                 prediction.shape[1]))),
                                      axis=1)
    cv2.imshow('Left:Input, center:Ground Truth, right:Prediction', image_example_gt)
    cv2.waitKey(0)


def plot_loss_accuracy(model_history):
    """
    Plot Loss and Accuracy (for training epoch)

    Args:
      model_history (cv2.Mat) - Image
      prediction (cv2.Mat) - Predicted output (mask)
      gt (cv2.Mat) - Expected output (mask)
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
