settings = dict()

settings['load_data'] = False
settings['zip_pathname'] = 'dataset/synthetic_sugarbeet_random_weeds'
settings['image_folder'] = settings['zip_pathname'] + '/rgb/'
settings['mask_folder'] = settings['zip_pathname'] + '/gt/'
settings['target_folder'] = settings['zip_pathname'] + '/train_test/'
settings['split_ratio'] = 0.9
settings['visualize_n_images'] = 5
settings['batch_size'] = 10
settings['training_height'] = 64
settings['training_width'] = 64
settings['training_step'] = 1000 / settings['batch_size']
settings['val_step'] = 1000 / settings['batch_size']
settings['epoch'] = 1

