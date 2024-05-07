import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
from matplotlib.colors import ListedColormap
import os
import json
from .utils import (get_implaus_thresh_t, 
                    get_implaus_thresh_conv, 
                    get_implaus_thresh_gaussian, 
                    get_implaus_thresh_t_boot_nonpivotal, 
                    get_implaus_thresh_t_boot_pivotal
                    )


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

    implausibilites = pd.read_csv(save_here_dir + 'implausibilities.csv')
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
    elif stats_dist_method == 'student-t_bootstrap_pivotal':
        cv = get_implaus_thresh_t_boot_pivotal(args, distances, variances)
    elif stats_dist_method == 'student-t_bootstrap_nonpivotal':
        cv = get_implaus_thresh_t_boot_nonpivotal(args)

    if '_pivotal' in stats_dist_method:
        print('Pivotal bootstraps obtained.')
    else:
        print(f'Threshold for 95th percentile: {round(cv,2)}')
    
    my_input_df = inputs_df.copy()
    my_input_df['implausibilities'] = implausibilites
    my_input_df['threshold'] = cv
    
    if type(cv) != list:
        cv = [cv]

    save_thresh_df = pd.DataFrame(cv,columns=['I_thresh'])
    save_thresh_df.to_csv(save_here_dir + 'implausibilityThreshold.csv', index=False)

    title = 'Strict bounds implausibilities'

    my_input_df['colors'] = my_input_df['implausibilities'] > my_input_df['threshold']
    custom_cmap = ListedColormap(['blue', 'red'])

    markersize_here = 1
    if num_points < 50000:
        markersize_here = 0.1
    else:
        markersize_here = 0.01

    for param in param_short_names:
        fig = plt.figure(facecolor='white',dpi=1200)
        
        plt.scatter(
            my_input_df[param],
            my_input_df['implausibilities'],
            alpha=1,
            s=markersize_here,
            c=my_input_df['colors'],
            cmap=custom_cmap
        )

        if not '_pivotal' in stats_dist_method:
            plt.axhline(
                cv,
                c='r',
                label = 'Implausibility Threshold'
            )
            plt.legend()

        plt.xlabel(param_dict[param], fontsize=8)
        plt.ylabel(r'$I(u^k)$', fontsize = 20)

        # setting yfloor
        if '_pivotal' in stats_dist_method:
            yfloor = min(my_input_df['implausibilities'])-(0.1)*np.mean(my_input_df['implausibilities'])
        else:
            yfloor = min(
                min(my_input_df['implausibilities'])-(0.1)*np.mean(my_input_df['implausibilities']),
                cv[0]
            )

        # setting yceiling
        if '_pivotal' in stats_dist_method:
            yceiling = max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities'])
        else:
            yceiling = max(
                max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities']),
                cv[0]
            )
        
        plt.ylim([yfloor,yceiling])

        plt.savefig(save_implaus_figs_dir + param, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
