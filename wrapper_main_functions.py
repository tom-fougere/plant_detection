import os
import tensorflow as tf
import cv2
from manage_dataset import load_data, split_data
from visualization import plot_images_with_masks
from data_augmentation import create_train_generator, create_validation_generator, my_image_mask_generator
from processing_functions import preprocessing_masks
from ae_models import fcn8, unet

# Global variables
nb_trained_images = 0


def prepare_data(settings):

    if settings['load_data']:
        load_data(settings['zip_pathname'])

    nb_trained_images = split_data(settings['image_folder'],
                                   settings['mask_folder'],
                                   settings['target_folder'],
                                   settings['split_ratio'])
    print('Number of trained images:', nb_trained_images)


def visualize_data(settings):

    list_files = os.listdir(settings['image_folder'])

    example_images = []
    example_masks = []
    for i_images in range(settings['visualize_n_images']):
        example_images.append(cv2.imread(settings['image_folder'] + list_files[i_images]))
        example_masks.append(cv2.imread(settings['mask_folder'] + list_files[i_images]))

    plot_images_with_masks(example_images, example_masks, scale_percent=0.5)


def learn_data(settings):
    # Generators
    train_image_generator, train_mask_generator = create_train_generator(settings['target_folder'],
                                                                         settings['batch_size'],
                                                                         (settings['training_height'], settings['training_width']),
                                                                         preprocessing_masks)
    val_image_generator, val_mask_generator = create_validation_generator(settings['target_folder'],
                                                                          settings['batch_size'],
                                                                          (settings['training_height'], settings['training_width']),
                                                                          preprocessing_masks)

    my_train_generator = my_image_mask_generator(train_image_generator, train_mask_generator)
    my_val_generator = my_image_mask_generator(val_image_generator, val_mask_generator)

    # Build the FCN8
    model = fcn8(settings['training_height'], settings['training_width'], 3, 1)

    # Compile your model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train your model here
    history = model.fit(my_train_generator,
                        steps_per_epoch=settings['training_step'],
                        epochs=settings['epoch'],
                        verbose=1,
                        validation_data=my_val_generator,
                        validation_steps=settings['val_step'])