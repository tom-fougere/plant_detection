import tensorflow as tf
import numpy as np
from cv2 import resize

from processing_functions import preprocessing_masks, convert_rgb_mask_to_1channel_mask
from manage_dataset import get_data
from data_augmentation import create_train_generator, create_validation_generator, my_image_mask_generator
from ae_models import fcn8, unet
from visualization import plot_predictions

HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 10
TARGET_FOLDER = "dataset/synthetic_sugarbeet_random_weeds/train_test/"
EPOCHS = 1

######################################
# DATA AUGMENTATION
######################################
train_image_generator, train_mask_generator = create_train_generator(TARGET_FOLDER,
                                                                     BATCH_SIZE,
                                                                     (HEIGHT, WIDTH),
                                                                     preprocessing_masks)
val_image_generator, val_mask_generator = create_validation_generator(TARGET_FOLDER,
                                                                      BATCH_SIZE,
                                                                      (HEIGHT, WIDTH),
                                                                      preprocessing_masks)
my_train_generator = my_image_mask_generator(train_image_generator, train_mask_generator)
my_val_generator = my_image_mask_generator(val_image_generator, val_mask_generator)

######################################
# MODEL
######################################
model = fcn8(HEIGHT, WIDTH, 3, 1, n_filters=32)
# model = unet(HEIGHT, WIDTH, 3, 1)
# print(model.summary())

model.load_weights('model/weights.h5')

######################################
# COMPILATION
######################################
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# steps = 1000 / BATCH_SIZE
# print('Steps:', steps)
#
# # Train your model here
# history = model.fit(my_train_generator,
#                     steps_per_epoch=steps,
#                     epochs=EPOCHS,
#                     verbose=1,
#                     validation_data=my_val_generator,
#                     validation_steps=steps)

######################################
# Prediction
######################################
img, mask = get_data(TARGET_FOLDER + 'test/images/img/', TARGET_FOLDER + 'test/masks/img/', id=1)

img = resize(img, (HEIGHT, WIDTH))
image = img / 255.

mask = preprocessing_masks(mask)
mask = resize(mask, (HEIGHT, WIDTH))
mask = convert_rgb_mask_to_1channel_mask(mask)

# Preprocessing image to get the right dim
image_tensor = tf.expand_dims(image, axis=0)

# Predict output with this image
prediction = model.predict(image_tensor)
pred_image = np.array(tf.squeeze(prediction, axis=0))

plot_predictions([img], [pred_image], third_overlay=True, scale_percent=0.6)


