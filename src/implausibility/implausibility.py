import numpy as np
import pandas as pd
import os
import json
import netCDF4 as nc
from concurrent.futures import ThreadPoolExecutor
from .utils import calculate_implausibility


def implausibilities(args):

    print('---------Implausibilities---------')

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']

    inputs_file_path = eval_params['emulator_inputs_file_path']
    inputs_df = pd.read_csv(inputs_file_path)
    num_variants = inputs_df.shape[0]

    # Read in necessary statistics
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    mle_variant = mle_df['parameter_set_num']

    # unpack values
    additional_variance = float(mle_df['variance_mle'])
    try:
        epsilon = float(mle_df['epsilon'])
    except:
        epsilon = 0

    if 'student-t' in stats_dist_method:
        nu = float(mle_df['nu'])
    else:
        nu = 0

    # get distances and variances data
    nc_file = nc.Dataset(save_here_dir + 'distances_variances.nc', 'r')

    # parallelization algo
    def execute_calculations():
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(calculate_implausibility, variant, nc_file, stats_dist_method, variance_mle = additional_variance, nu = nu, epsilon = epsilon) for variant in range(num_variants)]
            data_arr = [future.result() for future in futures]

            # Concatenate all predictions into a single DataFrame
            return data_arr

    implaus_arr = execute_calculations()


    # This does not pass when bootstrap is requested (stats_dist_method='student-t_bootstrap')
    if stats_dist_method == 'student-t':
        nu_opt = float(mle_df['nu'])
        my_variances_adjusted = my_variances_adjusted * ((nu_opt-2)/nu_opt)
    
    if 'epsilon' in mle_df.columns:
        my_distances = my_distances + float(mle_df['epsilon'])

    # Calculate Impluasibility quantities for every parameter set
    implausibilities = pd.DataFrame(implaus_arr, columns = ['variant', 'I'])
    # Save Implausibility values
    implausibilities.to_csv(save_here_dir + 'implausibilities.csv', index=False)

    best_param_set_num = implausibilities.sort_values(['I']).index[0]

    save_this = pd.DataFrame([nc_file['distances'][:,:,:,best_param_set_num].flatten(),nc_file['variances'][:,:,:,best_param_set_num].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'mostPlausibleDistsVaris.csv',index=False)
    save_this = pd.DataFrame([nc_file['distances'][:,:,:,mle_variant].flatten(),nc_file['variances'][:,:,:,mle_variant].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv',index=False)

    nc_file.close()

    return None