import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
from matplotlib.colors import ListedColormap
import os
import json
from tqdm import tqdm


def combined_frequentist_confidence_set(args):
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
    save_comb_implaus_figs_dir = save_here_dir + 'comb_implaus_figures/'
    inputs_file_path = eval_params['emulator_inputs_file_path']
    param_dict = eval_params['parameters_dictionary']
    run_dirs = eval_params['directories']

    implausibilites = pd.read_csv(save_here_dir + 'combinedImplausibilities.csv')
    threshold = pd.read_csv(save_here_dir + 'allThresholds.csv')
    inputs_df = pd.read_csv(inputs_file_path)

    try:
        inputs_df.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        None

    param_short_names = list(inputs_df.columns)

    cv = float(threshold['total'])
    print(f'Threshold for 95th percentile: {round(cv,2)}')
    
    my_input_df = inputs_df.copy()
    my_input_df['implausibilities'] = implausibilites['total']
    my_input_df['threshold'] = cv

    my_input_df['colors'] = my_input_df['implausibilities'] > my_input_df['threshold']
    custom_cmap = ListedColormap(['blue', 'red'])

    markersize_here = 0.1
    # if num_points < 50000:
    #     markersize_here = 0.1
    # else:
    #     markersize_here = 0.01

    progress_bar = tqdm(total=len(param_short_names)*(len(run_dirs)+1), desc="Progress")

    for param in param_short_names:
        fig = plt.figure(facecolor='white',dpi=1200)
        
        # plot implausibility points
        plt.scatter(
            my_input_df[param],
            my_input_df['implausibilities'],
            alpha=1,
            s=markersize_here,
            c=my_input_df['colors'],
            cmap=custom_cmap
        )

        # plot line for implausibility threshold
        plt.axhline(
            cv,
            c='r',
            label = 'Implausibility Threshold'
        )
        plt.legend()

        plt.xlabel(param_dict[param], fontsize=8)
        plt.ylabel(r'$I(u^k)$', fontsize = 20)

        # setting y axis min
        yfloor = min(
            min(my_input_df['implausibilities'])-(0.1)*np.mean(my_input_df['implausibilities']),
            cv
        )

        # setting y axis max
        yceiling = max(
            max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities']),
            cv
        )
        
        plt.ylim([yfloor,yceiling])

        plt.savefig(save_comb_implaus_figs_dir + param, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)
        progress_bar.update(1)

    plot_bounds = []
    for num, dir in enumerate(run_dirs):
        diff_df = implausibilites[str(num)] - float(threshold[str(num)])
        plot_bounds.append(min(diff_df))
        plot_bounds.append(max(diff_df))
    plot_bounds = [abs(i) for i in plot_bounds]

    for num, dir in enumerate(run_dirs):
        subfolder = dir.split('/')[-1]

        for param in param_short_names:
            # print(param)
            fig = plt.figure(facecolor='white',dpi=1200)
            
            # plot implausibility points
            plt.scatter(
                my_input_df[param],
                implausibilites[str(num)] - float(threshold[str(num)]),
                alpha=1,
                s=markersize_here,
                c=implausibilites[str(num)] > float(threshold[str(num)]),
                cmap=custom_cmap
            )

            # plot line for implausibility threshold
            plt.axhline(
                0,
                c='k',
                linestyle='--',
                linewidth=1,
                label = 'Implausibility Threshold'
            )
            # plt.legend()

            plt.xlabel(param_dict[param], fontsize=8)
            plt.ylabel(r'$I(u^k)$ - T', fontsize = 16)
            
            plt.ylim([-1*max(plot_bounds),max(plot_bounds)])

            plt.savefig(save_implaus_figs_dir + subfolder + '/' + param, dpi=300)
            # plt.cla()
            # plt.clf()
            plt.close(fig)
            progress_bar.update(1)
    progress_bar.close()
    

