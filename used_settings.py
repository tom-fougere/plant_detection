settings = dict()

settings['mode'] = 'evaluate'
settings['model'] = 'fcn8'
settings['unzip'] = False
settings['zip_pathname'] = 'dataset/synthetic_sugarbeet_random_weeds'
settings['image_folder'] = settings['zip_pathname'] + '/rgb/'
settings['mask_folder'] = settings['zip_pathname'] + '/gt/'
settings['target_folder'] = settings['zip_pathname'] + '/train_test/'
settings['split_ratio'] = 0.9
settings['visualize_n_images'] = 5
settings['batch_size'] = 10
settings['epochs'] = 5
settings['height'] = 256
settings['width'] = 256
settings['model_json'] = 'model.h5'
settings['model_weights'] = 'weights.h5'
settings['binary_threshold'] = 0.2

