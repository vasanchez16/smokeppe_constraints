import numpy as np
import pandas as pd
import os
import json


def implausibilities(args, my_distances, my_variances):

    print('---------Implausibilities---------')

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']

    # Read in necessary statistics
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    mle_param_set_num = int(mle_df['parameter_set_num'])
    additional_variance = mle_df['variance_mle'].values[0]
    my_variances_adjusted = my_variances + additional_variance


    # This does not pass when bootstrap is requested (stats_dist_method='student-t_bootstrap')
    if stats_dist_method == 'student-t':
        nu_opt = float(mle_df['nu'])
        my_variances_adjusted = my_variances_adjusted * ((nu_opt-2)/nu_opt)
    
    if 'epsilon' in mle_df.columns:
        my_distances = my_distances + float(mle_df['epsilon'])

    # Calculate Impluasibility quantities for every parameter set
    implausibilities = np.sqrt(np.nansum((my_distances ** 2) / my_variances_adjusted, axis = (0,1,2)))
    implausibilities = pd.DataFrame(implausibilities)
    # Save Implausibility values
    implausibilities.to_csv(save_here_dir + 'implausibilities.csv', index=False)

    best_param_set_num = implausibilities.sort_values([0]).index[0]
    save_this = pd.DataFrame([my_distances[:,:,:,best_param_set_num].flatten(),my_variances[:,:,:,best_param_set_num].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'mostPlausibleDistsVaris.csv',index=False)
    save_this = pd.DataFrame([my_distances[:,:,:,mle_param_set_num].flatten(),my_variances[:,:,:,mle_param_set_num].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv',index=False)
    return
