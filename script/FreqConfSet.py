import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
from matplotlib.colors import ListedColormap
import os
import json
from src.inference.implausibility import get_implaus_thresh_t, get_implaus_thresh_conv, get_implaus_thresh_gaussian

def FreqConfSet(args):
    """
    Notes here
    """
    print('---------FreqConfSet---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    save_figs_dir = save_here_dir + 'figures/'
    inputs_file_path = eval_params['emulator_inputs_file_path']
    param_dict = eval_params['parameters_dictionary']
    stats_dist_method = eval_params['stats_distribution_method']

    implausibilites = pd.read_csv(save_here_dir + 'implausibilities.csv', index_col=0)
    inputs_df = pd.read_csv(inputs_file_path, index_col=0)
    obs_df = pd.read_csv(save_here_dir + 'outliers.csv', index_col=0)
    param_short_names = list(inputs_df.columns)
    
    num_points = sum(~obs_df['missing'] & ~obs_df['outlier'])

    if stats_dist_method == 'convolution':
        cv = get_implaus_thresh_conv(args)
    elif stats_dist_method == 'student-t':
        cv = get_implaus_thresh_t(args,num_points)
    elif stats_dist_method == 'gaussian':
        cv = get_implaus_thresh_gaussian(args)
    print(f'Threshold for 95th percentile: {cv}')
    
    save_thresh_df = pd.DataFrame([cv],columns=['I_thresh'])
    save_thresh_df.to_csv(save_here_dir + 'implausibilityThreshold.csv', index=False)

    my_input_df = inputs_df.copy()
    my_input_df['implausibilities'] = implausibilites
    title = 'Strict bounds implausibilities'

    my_input_df['colors'] = (my_input_df['implausibilities']>cv)
    custom_cmap = ListedColormap(['blue', 'red'])

    for param in param_short_names:
        fig = plt.figure(facecolor='white',dpi=1200)
        

        plt.scatter(
            my_input_df[param],
            my_input_df['implausibilities'],
            alpha=1,
            s=0.01,
            c=my_input_df['colors'],
            cmap=custom_cmap
        )
        plt.axhline(
            cv,
            c='r',
            label = 'Implausibility Threshold'
        )

        plt.xlabel(param_dict[param], fontsize=8)
        plt.ylabel(r'$I(u^k)$', fontsize = 20)
        plt.ylim([min(my_input_df['implausibilities'])-(0.1)*np.mean(my_input_df['implausibilities']),max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities'])])
        plt.legend()

        plt.savefig(save_figs_dir + param, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
