import os
import cv2
from tensorflow.keras.models import model_from_json

from processing_functions import preprocessing_masks, postprocessing_masks_prediction, postprocessing_masks_rgb, postprocessing_images
from data_augmentation import create_train_generator, create_validation_generator, my_image_mask_generator
from performance import mean_class_wise_metrics
from ae_models import fcn8, unet

from used_settings import *

# Set some parameters
mode = settings['mode']
IMAGE_HEIGHT = settings['height']
IMAGE_WIDTH = settings['width']
BS = settings['batch_size']
epochs = settings['epochs']
ae_model = settings['model']
images_folder_path = settings['target_folder']

# %%
if mode == 'prepare_data':

    from manage_dataset import load_data, split_data
    from visualization import plot_images_with_masks

    # %%
    # Split data into train/val folders
    #

    if settings['unzip']:
        load_data(settings['zip_pathname'])

    nb_trained_images = split_data(settings['image_folder'],
                                   settings['mask_folder'],
                                   settings['target_folder'],
                                   settings['split_ratio'])
    print('Number of trained images:', nb_trained_images)

    # %%
    # Display some images
    #

    list_files = os.listdir(settings['image_folder'])

    example_images = []
    example_masks = []
    for i_images in range(settings['visualize_n_images']):
        example_images.append(cv2.imread(settings['image_folder'] + list_files[i_images]))
        example_masks.append(cv2.imread(settings['mask_folder'] + list_files[i_images]))

    plot_images_with_masks(example_images, example_masks, scale_percent=0.5)


elif mode == 'train':

    # %%
    # Prepare model for training
    #

    # Define Network and compile model
    if ae_model == 'fcn8':
        model = fcn8(IMAGE_HEIGHT, IMAGE_WIDTH, 3, 1)
    elif ae_model == 'unet':
        model = unet(IMAGE_HEIGHT, IMAGE_WIDTH, 3, 1)

    # Data augmentation for training and validation sets
    train_image_generator, train_mask_generator = create_train_generator(images_folder_path,
                                                                         BS,
                                                                         (IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                         preprocessing_masks)
    val_image_generator, val_mask_generator = create_validation_generator(images_folder_path,
                                                                          BS,
                                                                          (IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                          preprocessing_masks)
    my_train_generator = my_image_mask_generator(train_image_generator, train_mask_generator)
    my_val_generator = my_image_mask_generator(val_image_generator, val_mask_generator)

    # %%
    # Train the network
    #

    # Compile your model
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    H = model.fit(my_train_generator,
                  steps_per_epoch=train_image_generator.samples // BS,
                  validation_data=my_val_generator,
                  validation_steps=val_image_generator.samples // BS,
                  epochs=epochs,
                  max_queue_size=BS * 2,
                  verbose=1)

    # %%
    # Save the model
    #

    # serialize model to JSON
    model_json = model.to_json()
    with open('data/' + settings['model_json'], 'w') as json_file:
        json_file.write(model_json)
    # Save weights
    model.save_weights('data/' + settings['model_weights'])
    print("Saved model to disk")

# %%

elif mode == 'evaluate':

    # load json and create model
    json_file = open('data/' + settings['model_json'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('data/' + settings['model_weights'])
    print("Loaded model from disk")

    # Generate validation generator
    val_image_generator, val_mask_generator = create_validation_generator(images_folder_path,
                                                                          BS,
                                                                          (IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                          preprocessing_masks)
    my_val_generator = my_image_mask_generator(val_image_generator, val_mask_generator)

    for (images, masks) in my_val_generator:

        # Compute prediction
        predictions = model.predict(images)
        average_iou, average_dice = mean_class_wise_metrics(masks, predictions, 1)
        break

    print('Average IoU:', average_iou * 100)
    print('Average Dice:', average_dice * 100)


elif mode == 'visualize':

    import tensorflow as tf
    from visualization import plot_predictions

    # load json and create model
    json_file = open('data/' + settings['model_json'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('data/' + settings['model_weights'])
    print("Loaded model from disk")

    # Visualize  predictions
    list_files = os.listdir(settings['image_folder'])

    images = []
    masks = []
    for i_images in range(settings['visualize_n_images']):
        image = cv2.imread(settings['image_folder'] + list_files[i_images])
        mask = cv2.imread(settings['mask_folder'] + list_files[i_images])

        images.append(postprocessing_images(image, new_size=(IMAGE_HEIGHT, IMAGE_WIDTH)))
        masks.append(postprocessing_masks_rgb(mask, new_size=(IMAGE_HEIGHT, IMAGE_WIDTH)))

    # Make predictions
    images_tensor = tf.convert_to_tensor(images)
    raw_preds = model.predict(images_tensor)

    predictions = []
    for i_images in range(settings['visualize_n_images']):
        pred = postprocessing_masks_prediction(raw_preds[i_images],
                                               new_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        predictions.append(pred)

    # x = concatenate_several_images_masks_predictions(images, masks, predictions)
    plot_predictions(images, masks, predictions)
