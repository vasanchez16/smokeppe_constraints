import pandas as pd
import numpy as np
import netCDF4 as nc
import multiprocessing as mp
import scipy
import scipy.stats
from scipy.optimize import minimize_scalar, minimize
import sys
import os
import json
from tqdm import tqdm
from src.storage.utils import save_dataset
sys.path.append(os.getcwd())


def mle_gauss_old(args, distances, variances, num_variants):
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']
    
    # Define function to be used in minimize scalar
    def minus_log_l(d):
        sigma_opt = d[0]
        if len(init_vals) > 1:
            epsilon = d[1]
        else:
            epsilon = 0
        # get dists for one param set
        dists = distances[:,:,:,param_set].flatten()
        varis = variances[:,:,:,param_set].flatten()
        # Log likelihood, to be maximized
        term1 = np.nansum(np.log(varis + sigma_opt**2)) #get all the gspts for emulation variant
        term2 = np.nansum(np.power(dists + epsilon, 2) / (varis + sigma_opt**2))
        return 0.5 * (term1 + term2)

    # Run minimize scalar for each parameter set
    max_l_for_us = []
    sigma_sqr_terms = []
    epsilon_terms = []
    progress_bar = tqdm(total=num_variants, desc="Progress")
    for u in range(num_variants):
        param_set = u
        x_0 = init_vals
        if len(init_vals) > 1:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])])
            epsilon_terms.append(res.x[1])
        else:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0])])
        max_l_for_us.append(-res.fun)
        sigma_sqr_terms.append(res.x[0]**2)
        progress_bar.update(1)
    progress_bar.close()

    if len(init_vals) > 1:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,epsilon_terms], index = ['log_L', 'sigma_sqr', 'epsilon']).transpose()
    else:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms], index = ['log_L', 'sigma_sqr']).transpose()
    save_dataset(all_mle, save_here_dir + 'all_mle.csv')

    # Find parameter set that gives the max likelihood
    u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # Use this parmeter set to get the model discrep term at that parameter set
    param_set = u_mle
    x_0 = init_vals
    if len(init_vals) > 1:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])]).x #val for model discrep term
    else:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0])]).x #val for model discrep term

    sigma = dec_vars[0]
    sigma_sqr = sigma**2
    column_names = ['parameter_set_num', 'variance_mle']
    optimized_vals = [u_mle, sigma_sqr]

    if len(init_vals) > 1:
        epsilon = dec_vars[1]
        column_names = column_names + ['epsilon']
        optimized_vals = optimized_vals + [epsilon]

    return optimized_vals, column_names


################################################
################################################
################################################

# Moved outside mle_t function
def run_opt(variant_adj, variant, nc_file_path, init_vals, bnds):
    # Open the NetCDF file within the process
    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        dists_here = open_nc_file['distances'][:,:,:,variant_adj].flatten()
        varis_here = open_nc_file['variances'][:,:,:,variant_adj].flatten()
    
    x_0 = init_vals
    if len(init_vals) > 1:
        res = minimize(minus_log_l_with_epsilon, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='Nelder-Mead')
    else:    
        res = minimize(minus_log_l, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='Nelder-Mead')

    res_arr = [variant] + [res_var**2 if i == 0 else res_var for i, res_var in enumerate(res.x)]
    res_arr.append(-res.fun)

    if variant % 5000 == 0:
        print(f'Variant: {variant}')

    res_arr = [str(el) for el in res_arr]
    
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

def mle_gauss(args, num_variants):
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
    path_to_data_files = save_here_dir + 'dists_varis_data/'

    progress_file_path =  save_here_dir + 'mle_progress.txt'

    if not os.path.exists(progress_file_path):
        with open(progress_file_path, 'w') as new_prog_file:
            new_prog_file.write('')

    data_files = os.listdir(path_to_data_files)
    data_files = sorted(data_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    for file in data_files:

        # tracking completed files
        with open(progress_file_path,'r') as prog_file:
            contents = prog_file.read()
        current_completed_data_files = contents.split('\n')
        if file in current_completed_data_files:
            continue
        with open(progress_file_path, 'a') as prog_file:
            prog_file.write('\n' + file)

        # Parallelization
        nc_file_path = path_to_data_files + file

        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data
        min_variant = min(variants)
        variants_adj = [v - min_variant for v in variants]

        def execute_calculations():
            with mp.Pool(processes=mp.cpu_count()) as pool:
                futures = [pool.apply_async(run_opt, args=(variant_adj, variant, nc_file_path, init_vals, bnds)) for variant_adj, variant in zip(variants_adj, variants)]

                data_arr = [future.get() for future in futures]
            return data_arr

        mle_arr = execute_calculations()

        # Save data
        cols_here = ['parameter_set_num', 'variance_mle', 'epsilon', 'log_L'] if len(init_vals) > 1 else ['parameter_set_num', 'variance_mle', 'log_L']
        if not os.path.exists(save_here_dir + 'all_mle.csv'):
            with open(save_here_dir + 'all_mle.csv', 'w') as mle_file:
                mle_file.write(','.join(cols_here) + '\n')

        mle_arr = [','.join(i) for i in mle_arr]
        mle_results = '\n'.join(mle_arr) + '\n'

        with open(save_here_dir + 'all_mle.csv', 'a') as mle_file:
            mle_file.write(mle_results)
        print(f'{file} complete.')

    all_mle = pd.read_csv(save_here_dir + 'all_mle.csv')
    likelihood_vals = all_mle['log_L'].values
    mle_param_ind = np.argmax(likelihood_vals)
    optimized_vals = all_mle.iloc[mle_param_ind,:]


    return optimized_vals, cols_here
