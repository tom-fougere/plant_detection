import os
import random
import cv2
from shutil import copyfile


# Function to create all directories from a string path
def make_directory(fullpath):
    splitted_data = fullpath.split('/')

    new_dir = []
    current_dir = '.'

    for folder in splitted_data:
        current_dir = current_dir + '/' + folder
        if not os.path.isdir(current_dir + '/'):
            new_dir.append(current_dir)
            os.mkdir(current_dir)

    return new_dir


# Write a python function called split_data which takes
# a FOLDER_IMAGES directory containing the images
# a FOLDER_MASKS directory containing the masks
# a TARGET_FOLDER directory where the files will be copied to
# the TARGET_FOLDER directory will contain 2 sub-folders train and test with the data splitted
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SPLIT SIZE is .9
# Then 90% of the images will be copied to the TARGET_FOLDER/train dir
# and 10% of the images will be copied to the TARGET_FOLDER/test dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
def split_data(FOLDER_IMAGES, FOLDER_MASKS, TARGET_FOLDER, SPLIT_SIZE):
    TRAINING_FOLDER_NAME_IMAGES = TARGET_FOLDER + '/train/images/img/'
    TRAINING_FOLDER_NAME_MASKS = TARGET_FOLDER + '/train/masks/img/'
    TESTING_FOLDER_NAME_IMAGES = TARGET_FOLDER + '/test/images/img/'
    TESTING_FOLDER_NAME_MASKS = TARGET_FOLDER + '/test/masks/img/'

    # Create directories
    make_directory(TRAINING_FOLDER_NAME_IMAGES)
    make_directory(TRAINING_FOLDER_NAME_MASKS)
    make_directory(TESTING_FOLDER_NAME_IMAGES)
    make_directory(TESTING_FOLDER_NAME_MASKS)

    # Remove all data in TRAINING and TESTING dir
    for i_file in os.listdir(TRAINING_FOLDER_NAME_IMAGES):
        os.remove(TRAINING_FOLDER_NAME_IMAGES + i_file)
    for i_file in os.listdir(TRAINING_FOLDER_NAME_MASKS):
        os.remove(TRAINING_FOLDER_NAME_MASKS + i_file)
    for i_file in os.listdir(TESTING_FOLDER_NAME_IMAGES):
        os.remove(TESTING_FOLDER_NAME_IMAGES + i_file)
    for i_file in os.listdir(TESTING_FOLDER_NAME_MASKS):
        os.remove(TESTING_FOLDER_NAME_MASKS + i_file)

    dataset = []

    # Check for z zero file length
    for i_file in os.listdir(FOLDER_IMAGES):
        data = i_file
        if os.path.getsize(FOLDER_IMAGES + data) > 0:
            dataset.append(i_file)
        else:
            print('Skipped ' + i_file)
            print('Invalid file size! i.e Zero length.')

    # Number of files
    nb_files = len(dataset)
    nb_files_training = int(nb_files * SPLIT_SIZE)
    nb_files_testing = nb_files - nb_files_training

    # Shuffle dataset
    shuffled_dataset = random.sample(dataset, len(dataset))

    # Copy files
    for i_num, i_file in enumerate(shuffled_dataset):
        if i_num < nb_files_training:
            new_path_images = TRAINING_FOLDER_NAME_IMAGES + i_file
            new_path_masks = TRAINING_FOLDER_NAME_MASKS + i_file
        else:
            new_path_images = TESTING_FOLDER_NAME_IMAGES + i_file
            new_path_masks = TESTING_FOLDER_NAME_MASKS + i_file

        copyfile(FOLDER_IMAGES + i_file, new_path_images)
        copyfile(FOLDER_MASKS + i_file, new_path_masks)

    return nb_files_training


def get_data(image_path, mask_path, id=1):
    list_files = os.listdir(image_path)

    image = cv2.imread(image_path + list_files[id])
    mask = cv2.imread(mask_path + list_files[id])

    return image, mask

