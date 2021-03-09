import tensorflow as tf
import numpy as np

from processing_functions import preprocessing_masks
from data_augmentation import create_train_generator, create_validation_generator, my_image_mask_generator

HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 10
TARGET_FOLDER = "dataset/synthetic_sugarbeet_random_weeds/train_test/"
EPOCHS = 1


def build_generators(folder_path, train_batch_size, val_batch_size, height, width):
    """
    Build the Image data generators for image augmentation

    :param folder_path: path of the folder containing the images/masks
    :param train_batch_size: Batch size for the training generator
    :param val_batch_size:  Batch size for the validation generator
    :param height: Height of the image to generate
    :param width: Width of the image to generate
    :return my_train_generator, my_val_generator: two image data generators
    """
    train_image_generator, train_mask_generator = create_train_generator(folder_path,
                                                                         train_batch_size,
                                                                         (height, width),
                                                                         preprocessing_masks)
    val_image_generator, val_mask_generator = create_validation_generator(folder_path,
                                                                          val_batch_size,
                                                                          (height, width),
                                                                          preprocessing_masks)
    my_train_generator = my_image_mask_generator(train_image_generator, train_mask_generator)
    my_val_generator = my_image_mask_generator(val_image_generator, val_mask_generator)

    return my_train_generator, my_val_generator


def fit_model(model, generators, training_step, val_step, epoch=1):
    """
    Fit the model and the data

    :param model: network model
    :param generators: List of train/val generators
    :param training_step: Step for the training
    :param val_step: Step for the validation
    :param epoch: Number of epochs
    :return: History of the learning vs epochs
    """
    # Compile your model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train your model here
    history = model.fit(generators[0],
                        steps_per_epoch=training_step,
                        epochs=epoch,
                        verbose=1,
                        validation_data=generators[1],
                        validation_steps=val_step)

    return history



