import pandas as pd
import numpy as np
import netCDF4 as nc
import multiprocessing as mp
import scipy.special
from scipy.optimize import minimize
import sys
import os
import json
from src.storage.utils import save_dataset
sys.path.append(os.getcwd())


def minus_log_l(d, dists, varis):
    """Negative log-likelihood for the student-t distribution without a bias term."""
    sigma_opt = d[0]
    nu_opt = d[1]

    coeff = scipy.special.gamma((nu_opt + 1) / 2) / \
            (scipy.special.gamma(nu_opt / 2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))
    factor2 = 1 + (dists**2) / ((varis + sigma_opt**2) * (nu_opt - 2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    return -np.nansum(np.log(f_t))


def minus_log_l_with_epsilon(d, dists, varis):
    """Negative log-likelihood for the student-t distribution with a bias term (epsilon)."""
    sigma_opt = d[0]
    nu_opt = d[1]
    epsilon = d[2]

    coeff = scipy.special.gamma((nu_opt + 1) / 2) / \
            (scipy.special.gamma(nu_opt / 2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))
    factor2 = 1 + ((dists + epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt - 2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    return -np.nansum(np.log(f_t))


def run_opt(variant_adj, variant, nc_file_path, init_vals, bnds):
    """
    Run bounded MLE optimization for a single emulator variant.

    Reads distances and variances from the NetCDF file at the given variant index,
    then minimizes the negative log-likelihood. The first decision variable (sigma)
    is squared in the result to convert it to variance_mle.
    """
    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        dists_here = open_nc_file['distances'][:, :, :, variant_adj].flatten()
        varis_here = open_nc_file['variances'][:, :, :, variant_adj].flatten()

    x_0 = init_vals
    if len(init_vals) > 2:
        res = minimize(minus_log_l_with_epsilon, x_0, args=(dists_here, varis_here),
                       bounds=[tuple(b) for b in bnds], method='L-BFGS-B')
    else:
        res = minimize(minus_log_l, x_0, args=(dists_here, varis_here),
                       bounds=[tuple(b) for b in bnds], method='L-BFGS-B')

    if variant % 5000 == 0:
        print(f'Variant: {variant}')

    res_arr = [variant] + [val**2 if i == 0 else val for i, val in enumerate(res.x)]
    res_arr.append(-res.fun)
    return [str(el) for el in res_arr]


def mle_gauss(args, num_variants):
    """
    Run MLE analysis using the student-t distribution with multiprocessing.

    Iterates over pre-computed distances/variances NetCDF files in the run's
    dists_varis_data/ directory, runs bounded L-BFGS-B optimization in parallel
    for each emulator variant, and appends results to all_mle.csv. A progress
    file (mle_progress.txt) tracks completed data files so the run can resume
    safely if interrupted.

    Returns the row of all_mle.csv for the maximum-likelihood variant, along
    with the column names.
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    path_to_data_files = save_here_dir + 'dists_varis_data/'
    progress_file_path = save_here_dir + 'mle_progress.txt'

    cols_here = ['parameter_set_num', 'variance_mle', 'nu', 'epsilon', 'log_L'] if len(init_vals) > 2 \
        else ['parameter_set_num', 'variance_mle', 'nu', 'log_L']

    if not os.path.exists(progress_file_path):
        with open(progress_file_path, 'w') as f:
            f.write('')

    data_files = sorted(
        os.listdir(path_to_data_files),
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )

    for file in data_files:
        with open(progress_file_path, 'r') as prog_file:
            completed = prog_file.read().split('\n')
        if file in completed:
            continue

        with open(progress_file_path, 'a') as prog_file:
            prog_file.write('\n' + file)

        nc_file_path = path_to_data_files + file
        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data
        min_variant = min(variants)
        variants_adj = [v - min_variant for v in variants]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            futures = [
                pool.apply_async(run_opt, args=(va, v, nc_file_path, init_vals, bnds))
                for va, v in zip(variants_adj, variants)
            ]
            mle_arr = [f.get() for f in futures]

        if not os.path.exists(save_here_dir + 'all_mle.csv'):
            with open(save_here_dir + 'all_mle.csv', 'w') as mle_file:
                mle_file.write(','.join(cols_here) + '\n')

        with open(save_here_dir + 'all_mle.csv', 'a') as mle_file:
            mle_file.write('\n'.join(','.join(row) for row in mle_arr) + '\n')

        print(f'{file} complete.')

    all_mle = pd.read_csv(save_here_dir + 'all_mle.csv')
    mle_param_ind = np.argmax(all_mle['log_L'].values)
    optimized_vals = all_mle.iloc[mle_param_ind, :]

    return optimized_vals, cols_here
