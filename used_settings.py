settings = dict()

settings['mode'] = 'visualize'  # prepare_data, train, evaluate, visualize
settings['model'] = 'fcn8'
settings['unzip'] = False
settings['zip_pathname'] = 'dataset/synthetic_sugarbeet_random_weeds'
settings['image_folder'] = settings['zip_pathname'] + '/rgb/'
settings['mask_folder'] = settings['zip_pathname'] + '/gt/'
settings['target_folder'] = settings['zip_pathname'] + '/train_test/'
settings['split_ratio'] = 0.9
settings['visualize_n_images'] = 5
settings['batch_size'] = 20
settings['learning_rate'] = 0.01
settings['epochs'] = 3
settings['height'] = 512
settings['width'] = 512
settings['model_json'] = 'model.h5'
settings['model_weights'] = 'weights.h5'
settings['binary_threshold'] = 0.2

