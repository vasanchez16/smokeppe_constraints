import numpy as np
import pandas as pd
import os
import json


def combined_implausibilities(args, run_dirs):

    print('---------Combined Implausibilities---------')

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    comb_implaus = pd.DataFrame()
    all_thresholds = pd.DataFrame()

    for num, dir_here in enumerate(run_dirs):
        if dir_here[-1] != '/':
            dir_here = dir_here + '/'
        
        # read in each implaus, thresh, and outs 
        implaus = pd.read_csv(dir_here + 'implausibilities.csv')
        thresh = pd.read_csv(dir_here + 'implausibilityThreshold.csv')
        outs = pd.read_csv(dir_here + 'outliers.csv')

        # calc total num of points for target variable
        num_points = sum(~outs['missing'] & ~outs['outlier'])
        scale_factor = num_points ** (-1/2)
        # scale_factor = 1

        # calc normalized implaus and thresh vals and save vals
        comb_implaus[str(num)] = implaus['0'] * scale_factor
        all_thresholds[str(num)] = [float(thresh['I_thresh']) * scale_factor]

    # sum all implaus for a total implaus
    comb_implaus['total'] = np.sqrt((comb_implaus.loc[:,:] ** 2).sum(axis=1))
    # comb_implaus['total'] = comb_implaus.sum(axis=1)

    # sum all thresholds for a total threshold
    all_thresholds['total'] = np.sqrt((all_thresholds.loc[:,:] ** 2).sum(axis=1))
    # all_thresholds['total'] = all_thresholds.sum(axis=1)

    # save data
    comb_implaus.to_csv(save_here_dir + 'combinedImplausibilities.csv',index=False)
    all_thresholds.to_csv(save_here_dir + 'allThresholds.csv',index=False)
    return
