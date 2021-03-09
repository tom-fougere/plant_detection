# Import wrapper
from wrapper_main_functions import prepare_data, visualize_data, learn_data, show_results, evaluate_model, save_weights

# Load settings
from used_settings import *

# Switch between 'prepare', 'learn' and 'evaluate'
s_processing_step = 'evaluate'

if s_processing_step == 'prepare':
    print('Step: PREPARING data...')
    prepare_data(settings)
elif s_processing_step == 'visualize':
    print('Step: VISUALIZING data...')
    visualize_data(settings)
elif s_processing_step == 'learn':
    print('Step: LEARNING...')
    learn_data(settings)
elif s_processing_step == 'evaluate':
    print('Step: EVALUATING...')
    evaluate_model(settings)
elif s_processing_step == 'results':
    print('Step: SHOWING RESULTS...')
    show_results(settings)
elif s_processing_step == 'save':
    print('Step: SAVING...')
    save_weights(settings)
else:
    print("'", s_processing_step, "' doesn't exist as a defined step")