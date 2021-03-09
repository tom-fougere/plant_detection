import os
import numpy as np
import tensorflow as tf
import cv2
from manage_dataset import load_data, split_data
from visualization import plot_images_with_masks, plot_predictions
from processing_functions import postprocessing_images, postprocessing_masks_rgb, postprocessing_masks_prediction, preprocessing_masks
from ae_models import fcn8, unet
from learning_functions import build_generators, fit_model
from performance import mean_class_wise_metrics

# Global variables
nb_trained_images = 0
model = fcn8(64, 64, 3, 1)


def prepare_data(settings):
    """
    Prepare the data (splitting into several folders)

    :param settings: dict containing all parameters
    :return: None
    """

    if settings['load_data']:
        load_data(settings['zip_pathname'])

    nb_trained_images = split_data(settings['image_folder'],
                                   settings['mask_folder'],
                                   settings['target_folder'],
                                   settings['split_ratio'])
    print('Number of trained images:', nb_trained_images)


def visualize_data(settings):
    """
    Visualize the some input/mask

    :param settings: dict containing all parameters
    :return: None
    """

    list_files = os.listdir(settings['image_folder'])

    example_images = []
    example_masks = []
    for i_images in range(settings['visualize_n_images']):
        example_images.append(cv2.imread(settings['image_folder'] + list_files[i_images]))
        example_masks.append(cv2.imread(settings['mask_folder'] + list_files[i_images]))

    plot_images_with_masks(example_images, example_masks, scale_percent=0.5)


def learn_data(settings):
    """
    Fit the data and the model

    :param settings: dict containing all parameters
    :return: None
    """

    # Generators
    my_train_generator, my_val_generator = build_generators(settings['target_folder'],
                                                            settings['batch_size'],
                                                            settings['batch_size'],
                                                            settings['training_height'],
                                                            settings['training_width'])

    # Build the FCN8
    model = fcn8(settings['training_height'], settings['training_width'], 3, 1)

    # Compile et run the model
    history = fit_model(model,
                        [my_train_generator, my_val_generator],
                        settings['training_step'],
                        settings['val_step'],
                        epoch=1)


def save_weights(settings):
    """
    Save the weight of the model

    :param settings: dict containing all parameters
    :return: None
    """
    if settings['weights'] == 'last':
        model.save_weights('weights.h5')
    else:
        model.save_weights(settings['weights'])


def load_weights(settings):
    """
    Load the weight of the model

    :param settings: dict containing all parameters
    :return: None
    """
    if settings['weights'] == 'last':
        model.load_weights('weights.h5')
    else:
        model.load_weights(settings['weights'])


def show_results(settings):
    """
    Display some result of the prediction

    :param settings: dict containing all parameters
    :return: None
    """

    list_files = os.listdir(settings['image_folder'])

    images = []
    masks = []
    for i_images in range(settings['visualize_n_images']):
        image = cv2.imread(settings['image_folder'] + list_files[i_images])
        mask = cv2.imread(settings['mask_folder'] + list_files[i_images])

        images.append(postprocessing_images(image, new_size=(settings['height'], settings['width'])))
        masks.append(postprocessing_masks_rgb(mask, new_size=(settings['height'], settings['width'])))

    # Make predictions
    images_tensor = tf.convert_to_tensor(images)
    result_model = fcn8(settings['height'], settings['width'], 3, 1)
    result_model.load_weights('weights.h5')
    raw_preds = result_model.predict(images_tensor)

    predictions = []
    for i_images in range(settings['visualize_n_images']):
        pred = postprocessing_masks_prediction(raw_preds[i_images],
                                               new_size=(settings['height'],
                                                         settings['width']))
        predictions.append(pred)

    # x = concatenate_several_images_masks_predictions(images, masks, predictions)
    plot_predictions(images, masks, predictions)


def evaluate_model(settings):
    """
    Evaluate the model (IoU and Dice scores) on validation data

    :param settings: dict containing all parameters
    :return: None
    """

    my_train_generator, my_val_generator = build_generators(settings['target_folder'],
                                                            settings['batch_size'],
                                                            126,
                                                            settings['training_height'],
                                                            settings['training_width'])

    for (images, masks) in my_val_generator:

        # Compute prediction
        predictions = model.predict(images)

        average_iou, average_dice = mean_class_wise_metrics(masks, predictions, 1)
        break

    print('Average IoU:', average_iou * 100)
    print('Average Dice:', average_dice * 100)