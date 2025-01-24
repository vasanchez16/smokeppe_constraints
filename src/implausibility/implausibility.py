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

    with open(save_here_dir + 'implausibilities.csv', 'w') as implaus_file:
        implaus_file.write('variant,I' + '\n')

    min_implaus_arr = []

    # get distances and variances data files
    data_files = os.listdir(path_to_data_files)
    data_files = sorted(data_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    num_processes = min(mp.cpu_count(), len(data_files))

    def execute_calculations():
        with mp.Pool(processes=6) as pool:
            futures = [pool.apply_async(calculate_implausibility, args=(path_to_data_files + file, stats_dist_method, additional_variance, nu, epsilon)) for file in data_files]
            data_arr = [future.get() for future in futures]
                        
        return data_arr
    
    many_implaus_arrs = execute_calculations()

    for chunk, file in zip(many_implaus_arrs, data_files):
        print(f'Appending Implaus for {file}...')
        chunk_str = [','.join(i) for i in chunk]
        chunk_str = '\n'.join(chunk_str) + '\n'

        best_variant_data_here = sorted(chunk, key=lambda x: float(x[-1]))[0]

        # saving minimum implaus
        if file == data_files[0]:
            min_implaus_arr = best_variant_data_here
        else:
            if best_variant_data_here[-1] < min_implaus_arr[-1]:
                min_implaus_arr = best_variant_data_here
            else:
                None

        with open(save_here_dir + 'implausibilities.csv', 'a') as implaus_file:
            implaus_file.write(chunk_str)
    
    best_param_set_num = int(min_implaus_arr[0])

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
    save_this = pd.DataFrame([nc_file['distances'][...,get_this_ind].flatten(),nc_file['variances'][...,get_this_ind].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'mostPlausibleDistsVaris.csv',index=False)
    nc_file.close()

    nc_file = nc.Dataset(path_to_data_files + data_files[mle_data_file_ind],'r')
    variants = nc_file['variant'][:].data
    min_variant = min(variants)
    get_this_ind = mle_variant - min_variant
    save_this = pd.DataFrame([nc_file['distances'][...,get_this_ind].flatten(),nc_file['variances'][...,get_this_ind].flatten()],index=['dists','varis']).transpose()
    save_this.to_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv',index=False)
    nc_file.close()

    return None