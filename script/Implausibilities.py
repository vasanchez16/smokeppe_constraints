import numpy as np
import pandas as pd
import os
import json

def Implausibilities(args, my_distances, my_variances):

    print('---------Implausibilities---------')

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    # Read in necessary statistics
    mle_df = pd.read_csv(save_here_dir + 'mle.csv',index_col=0)

    mle_param_set_num = int(mle_df.iloc[0,0])
    additional_variance = mle_df['variance_mle'].values[0]

    # print('Reading in dists...')
    # my_distances = pd.read_csv(save_here_dir + 'distances.csv',index_col=0)
    # print('Reading in varis...')
    # my_variances = pd.read_csv(save_here_dir + 'variances.csv',index_col=0)
    my_variances_adjusted = my_variances + additional_variance

    # Calculate Impluasibility quantities for every parameter set
    implausibilities = np.sqrt(np.power(my_distances, 2).div(my_variances_adjusted).sum(axis=0))
    # Save Implausibility values
    implausibilities.to_csv(save_here_dir + 'implausibilities.csv', index=True)

    best_param_set_num = implausibilities.sort_values(['0']).index[0]
    save_this = pd.DataFrame([my_distances.iloc[:,best_param_set_num],my_variances.iloc[:,best_param_set_num]],index=['dists','varis']).transpose()
    save_this.to_csv('/ocean/projects/atm200005p/vsanchez/coarseGrainedOutputs/cdnc_100k/oneParamStats.csv',index=False)
    
    return