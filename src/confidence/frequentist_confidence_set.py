import pandas as pd
import numpy as np
import scipy
import scipy.stats
import math
from matplotlib.colors import ListedColormap
import os
import json
from .utils import (get_implaus_thresh_t, 
                    get_implaus_thresh_conv, 
                    get_implaus_thresh_gaussian,
                    get_implaus_thresh_t_boot,
                    get_implaus_thresh_gauss_boot)
from .viz import plot_constraint_1d, all_param_implaus


def frequentist_confidence_set(args, distances, variances):
    """
    Notes here
    """
    print('---------FreqConfSet---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    save_implaus_figs_dir = save_here_dir + 'implaus_figures/'
    inputs_file_path = eval_params['emulator_inputs_file_path']
    param_dict = eval_params['parameters_dictionary']
    stats_dist_method = eval_params['stats_distribution_method']

    implausibilities = pd.read_csv(save_here_dir + 'implausibilities.csv')
    inputs_df = pd.read_csv(inputs_file_path)
    obs_df = pd.read_csv(save_here_dir + 'outliers.csv')

    try:
        inputs_df.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        None

    param_short_names = list(inputs_df.columns)
    
    num_points = sum(~obs_df['missing'] & ~obs_df['outlier'])

    if stats_dist_method == 'convolution':
        cv = get_implaus_thresh_conv(args)
    elif stats_dist_method == 'student-t':
        cv = get_implaus_thresh_t(args,num_points)
    elif stats_dist_method == 'gaussian':
        cv = get_implaus_thresh_gaussian(args)
    elif stats_dist_method == 'gaussian_bootstrap':
        cv = get_implaus_thresh_gauss_boot(args)
    elif stats_dist_method == 'student-t_bootstrap':
        cv = get_implaus_thresh_t_boot(args)

    cv_raw = cv
    cv = cv / np.sqrt(num_points)
    print(f'Threshold for 95th percentile: {round(cv,2)}')

    implausibilities = implausibilities / np.sqrt(num_points)
    norm_implaus = pd.DataFrame(implausibilities)
    # Save norm Implausibility values
    norm_implaus.to_csv(save_here_dir + 'norm_implausibilities.csv', index=False)
    
    my_input_df = inputs_df.copy()
    my_input_df['implausibilities'] = implausibilities
    my_input_df['threshold'] = cv

    save_thresh_df = pd.DataFrame([cv, cv_raw],index=['I_thresh', 'raw_I_thresh']).transpose()
    save_thresh_df.to_csv(save_here_dir + 'implausibilityThreshold.csv', index=False)

    my_input_df['colors'] = my_input_df['implausibilities'] > my_input_df['threshold']
    custom_cmap = ListedColormap(['blue', 'red'])

    markersize_here = 1
    if num_points < 50000:
        markersize_here = 0.1
    else:
        markersize_here = 0.01

    # Load MLE
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    for param in param_short_names:
        plot_constraint_1d(my_input_df,
                           param,
                           cv,
                           save_implaus_figs_dir,
                           param_dict,
                           custom_cmap,
                           markersize_here,
                           mle_idx=mle_df['parameter_set_num'])
        
    all_param_implaus(
        my_input_df,
        cv,
        save_implaus_figs_dir,
        param_dict,
        custom_cmap,
        markersize_here*20,
        param_short_names,
        mle_idx=mle_df['parameter_set_num'])
