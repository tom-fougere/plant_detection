import os
from cv2 import imread, resize
import random
from numpy import asarray
from tensorflow.keras.models import model_from_json

from processing_functions import preprocessing_masks, binarize_image, convert_rgb_mask_to_1channel_mask
from data_augmentation import create_train_generator, create_validation_generator, my_image_mask_generator
from performance import mean_iou_dice_score_multiclass
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
        example_images.append(imread(settings['image_folder'] + list_files[i_images]))
        example_masks.append(imread(settings['mask_folder'] + list_files[i_images]))

    plot_images_with_masks(example_images, example_masks, scale_percent=0.5)


elif mode == 'train':

    # Select the model
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

    # Compile the model
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    H = model.fit(my_train_generator,
                  steps_per_epoch=train_image_generator.samples // BS,
                  validation_data=my_val_generator,
                  validation_steps=val_image_generator.samples // BS,
                  epochs=epochs,
                  max_queue_size=BS * 2,
                  verbose=1)

    # Save the model
    model_json = model.to_json()
    with open('model/' + settings['model_json'], 'w') as json_file:
        json_file.write(model_json)
    # Save weights
    model.save_weights('model/' + settings['model_weights'])
    print("Saved model to disk")


elif mode == 'evaluate':

    # Load the model
    json_file = open('model/' + settings['model_json'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights in the model
    model.load_weights('model/' + settings['model_weights'])
    print("Loaded model from disk")

    # Extract the dimension's input
    model_input_dim = model.layers[0].output_shape[0]

    # Read the images for the validation
    image_folder = settings['target_folder'] + 'test/images/img/'
    mask_folder = settings['target_folder'] + 'test/masks/img/'
    list_files = os.listdir(image_folder)

    images_list = []
    masks_list = []
    for i_images in range(len(list_files)):
        cur_image = imread(image_folder + list_files[i_images])
        cur_image = resize(cur_image, (model_input_dim[1], model_input_dim[2]))
        images_list.append(cur_image)

        cur_mask = imread(mask_folder + list_files[i_images])
        cur_mask = preprocessing_masks(cur_mask)
        cur_mask = resize(cur_mask, (model_input_dim[1], model_input_dim[2]))
        cur_mask = convert_rgb_mask_to_1channel_mask(cur_mask)
        masks_list.append(cur_mask)

    # Convert list of images into 4d array (and format them for model)
    images = asarray(images_list) / 255.

    # Predict
    predictions = model.predict(images)

    # Convert tensor into list of images
    predictions_list = [binarize_image(predictions[i], settings['binary_threshold'])
                        for i in range(predictions.shape[0])]

    # Compute the IoU and dice score
    average_iou, average_dice = mean_iou_dice_score_multiclass(masks_list, predictions_list, nb_classes=2)

    print('Average IoU:', average_iou * 100)
    print('Average Dice:', average_dice * 100)


elif mode == 'visualize':

    from visualization import plot_predictions

    # Load the model
    json_file = open('model/' + settings['model_json'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights in the model
    model.load_weights('model/' + settings['model_weights'])
    print("Loaded model from disk")

    # Extract the dimension's input
    model_input_dim = model.layers[0].output_shape[0]

    # Read the images of the validation folder
    list_files = os.listdir(settings['image_folder'])

    images_list = []
    masks_list = []
    masks = []
    # for i_images in range(settings['visualize_n_images']):
    for i_images in random.sample(range(len(list_files)), settings['visualize_n_images']):
        cur_mask = imread(settings['mask_folder'] + list_files[i_images])
        cur_mask = resize(cur_mask, (model_input_dim[1], model_input_dim[2]))
        masks_list.append(cur_mask)
        cur_image = imread(settings['image_folder'] + list_files[i_images])
        cur_image = resize(cur_image, (model_input_dim[1], model_input_dim[2]))
        images_list.append(cur_image)

    # Convert list of images into 4d array (and format them for model)
    images = asarray(images_list) / 255.

    # Predict
    predictions = model.predict(images)

    # Convert tensor into list of images
    predictions_list = [predictions[i] for i in range(predictions.shape[0])]
    # predictions_list = [binarize_image(predictions[i], settings['binary_threshold'])
    #                     for i in range(predictions.shape[0])]

    # Plot prediction
    plot_predictions(images_list, predictions_list, scale_percent=0.5, third_overlay=True)
