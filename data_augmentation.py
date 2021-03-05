import tensorflow as tf

from processing_functions import preprocessing_masks

SEED = 0


def create_train_generator(target_folder, batch_size, target_size, mask_preprocessing_function):

    train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                                                                          horizontal_flip=True,
                                                                          rotation_range=20,
                                                                          zoom_range=0.2,
                                                                          width_shift_range=0.1,
                                                                          height_shift_range=0.1)
    train_image_generator = train_image_datagen.flow_from_directory(target_folder + 'train/images',
                                                                    batch_size=batch_size,
                                                                    target_size=target_size,
                                                                    color_mode='rgb',
                                                                    class_mode=None,
                                                                    seed=SEED)

    train_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=mask_preprocessing_function,
                                                                         horizontal_flip=True,
                                                                         rotation_range=20,
                                                                         zoom_range=0.2,
                                                                         width_shift_range=0.1,
                                                                         height_shift_range=0.1)
    train_mask_generator = train_mask_datagen.flow_from_directory(target_folder + 'train/masks/',
                                                                  batch_size=batch_size,
                                                                  target_size=target_size,
                                                                  color_mode='grayscale',
                                                                  class_mode=None,
                                                                  seed=SEED)

    return train_image_generator, train_mask_generator


def create_validation_generator(target_folder, batch_size, target_size, mask_preprocessing_function):
    val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    val_image_generator = val_image_datagen.flow_from_directory(target_folder + 'test/images',
                                                                batch_size=batch_size,
                                                                target_size=target_size,
                                                                class_mode=None,
                                                                seed=SEED)

    val_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=mask_preprocessing_function)
    val_mask_generator = val_mask_datagen.flow_from_directory(target_folder + 'test/masks/',
                                                              batch_size=batch_size,
                                                              target_size=target_size,
                                                              class_mode=None,
                                                              seed=SEED)

    return val_image_generator, val_mask_generator


# Create custom generator for training images and masks
def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield img, mask
