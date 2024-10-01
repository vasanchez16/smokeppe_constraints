import pandas as pd
import numpy as np
import netCDF4 as nc
import multiprocessing as mp
import scipy
from scipy.special import gamma
import scipy.stats
from scipy.optimize import minimize_scalar, minimize
import sys
import os
import json
from tqdm import tqdm
from src.storage.utils import save_dataset
sys.path.append(os.getcwd())


# Moved outside mle_t function
def run_opt(variant, nc_file_path, init_vals, bnds):
    # Open the NetCDF file within the process
    # with nc.Dataset(nc_file_path, 'r') as open_nc_file:
    #     dists_here = open_nc_file['distances'][:,:,:,variant].flatten()
    #     varis_here = open_nc_file['variances'][:,:,:,variant].flatten()
    if variant == 0:
        print('getting data...')
    dists_here = open_nc_file['distances'][:,:,:,variant].flatten()
    varis_here = open_nc_file['variances'][:,:,:,variant].flatten()
    if variant == 0:
        print('done getting data.')
    x_0 = init_vals
    if len(init_vals) > 2:
        res = minimize(minus_log_l_with_epsilon, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='Nelder-Mead')
    else:    
        res = minimize(minus_log_l, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='Nelder-Mead')

    res_arr = [variant] + [res_var**2 if i == 0 else res_var for i, res_var in enumerate(res.x)]
    res_arr.append(-res.fun)

    if variant % 10 == 0:
        print(f'Variant: {variant}')
    
    return res_arr

def minus_log_l(d, dists, varis):
    sigma_opt = d[0]
    nu_opt = d[1]
    epsilon = 0

    coeff = scipy.special.gamma((nu_opt + 1) / 2) / \
            (scipy.special.gamma(nu_opt / 2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))
    factor2 = 1 + ((dists + epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt-2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    log_Li = np.log(f_t)
    log_likelihood = np.nansum(log_Li)
    return -1 * log_likelihood

def minus_log_l_with_epsilon(d, dists, varis):
    sigma_opt = d[0]
    nu_opt = d[1]
    epsilon = d[2]

    coeff = scipy.special.gamma((nu_opt + 1) / 2) / \
            (scipy.special.gamma(nu_opt / 2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))
    factor2 = 1 + ((dists + epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt-2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    log_Li = np.log(f_t)
    log_likelihood = np.nansum(log_Li)
    return -1 * log_likelihood

def mle_t(args, num_variants):
    """
    MLE analysis using the student-t distribution approximation.
    """

    # Load parameters from input file
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)
    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    nc_file_path = save_here_dir + 'distances_variances.nc'

    # Parallelization
    print('reading...')
    global open_nc_file 
    open_nc_file = nc.Dataset(nc_file_path,'r')
    print('done reading')
    
    def execute_calculations():
        with mp.Pool(processes=mp.cpu_count()) as pool:
            futures = [pool.apply_async(run_opt, args=(variant, nc_file_path, init_vals, bnds)) for variant in range(num_variants)]

            data_arr = [future.get() for future in futures]
        return data_arr

    mle_arr = execute_calculations()
    open_nc_file.close()
    # Save data
    cols_here = ['parameter_set_num', 'variance_mle', 'nu', 'epsilon', 'log_L'] if len(init_vals) > 2 else ['parameter_set_num', 'variance_mle', 'nu', 'log_L']
    all_mle_df = pd.DataFrame(mle_arr, columns=cols_here)
    save_dataset(all_mle_df, save_here_dir + 'all_mle.csv')

    # Sort by log likelihood (last column)
    mle_arr = sorted(mle_arr, key=lambda x: x[-1], reverse=True)

    # Get optimized values
    optimized_vals = [mle_arr[0][0], mle_arr[0][1], mle_arr[0][2]]
    if len(init_vals) > 2:
        optimized_vals.append(mle_arr[0][3])

    optimized_vals.append(mle_arr[0][-1])

    return optimized_vals, cols_here