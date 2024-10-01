import numpy as np
import pandas as pd
import os
import json
import netCDF4 as nc
import multiprocessing as mp
from .utils import calculate_implausibility


def implausibilities(args):

    print('---------Implausibilities---------')

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']
    path_to_data_files = save_here_dir + 'dists_varis_data/'

    inputs_file_path = eval_params['emulator_inputs_file_path']
    inputs_df = pd.read_csv(inputs_file_path)
    num_variants = inputs_df.shape[0]

    # Read in necessary statistics
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    mle_variant = int(mle_df['parameter_set_num'])

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

    # get distances and variances data files
    data_files = os.listdir(path_to_data_files)

    # parallelization algo
    def execute_calculations():
        with mp.Pool(processes=mp.cpu_count()) as pool:
            futures = [pool.apply_async(calculate_implausibility, args = (path_to_data_files + file, stats_dist_method, additional_variance, nu, epsilon)) for file in data_files]
            data_arr = [future.get() for future in futures]

        # Concatenate all predictions into a single DataFrame
        return data_arr

    implaus_arr = execute_calculations()

    implaus_arr = [np.array(i) for i in implaus_arr]
    implaus_arr = np.concatenate(implaus_arr)
    print(implaus_arr[0])
    print(len(implaus_arr[0]))
    print(type(implaus_arr[0][0]))
    print(implaus_arr[0][1])
    # error
    # Calculate Impluasibility quantities for every parameter set
    implausibilities = pd.DataFrame(implaus_arr, columns = ['variant', 'I'])
    implausibilities.sort_values(['variant'], ignore_index=True, inplace=True)
    implausibilities.get('variant').apply(lambda x: int(x))
    # Save Implausibility values
    implausibilities.to_csv(save_here_dir + 'implausibilities.csv', index=False)

    best_param_set_num = implausibilities.sort_values(['I']).index[0]

    data_files_nums = [int(f.split('_')[2].split('.nc')[0]) for f in data_files]
    diff_nums = np.array([t - mle_variant for t in data_files_nums])
    closest_num = min(diff_nums[diff_nums >= 0])
    mle_data_file_ind = np.argmax(diff_nums == closest_num)
    diff_nums = np.array([t - best_param_set_num for t in data_files_nums])
    closest_num = min(diff_nums[diff_nums >= 0])
    best_param_data_file_ind = np.argmax(diff_nums == closest_num)
    
    nc_file = nc.Dataset(path_to_data_files + data_files[best_param_data_file_ind],'r')
    variants = nc_file['variant'][:].data
    min_variant = min(variants)
    get_this_ind = best_param_set_num - min_variant
    save_this = pd.DataFrame([nc_file['distances'][:,:,:,get_this_ind].flatten(),nc_file['variances'][:,:,:,get_this_ind].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'mostPlausibleDistsVaris.csv',index=False)
    nc_file.close()

    nc_file = nc.Dataset(path_to_data_files + data_files[mle_data_file_ind],'r')
    variants = nc_file['variant'][:].data
    min_variant = min(variants)
    get_this_ind = mle_variant - min_variant
    save_this = pd.DataFrame([nc_file['distances'][:,:,:,get_this_ind].flatten(),nc_file['variances'][:,:,:,get_this_ind].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv',index=False)
    nc_file.close()

    return None