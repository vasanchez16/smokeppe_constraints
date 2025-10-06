import pandas as pd
import numpy as np
import netCDF4 as nc
import scipy
from scipy.special import gamma
import scipy.stats
from scipy.optimize import minimize_scalar, minimize
import sys
import os
import json
from tqdm import tqdm
from src.storage.utils import get_mle_columns, save_mle_to_nc
sys.path.append(os.getcwd())

from mpi4py import MPI

COMM = MPI.COMM_WORLD
CRANK = COMM.Get_rank()
CSIZE = COMM.Get_size()
CROOT = 0

FILE_PROGRESS_TAG = 11

"""
Optimization functions
"""

def minus_log_l(d, dists, varis):
    sigma_opt = d[0]
    nu_opt = d[1]
    epsilon = 0

    coeff = (scipy.special.gamma((nu_opt + 1) / 2) / scipy.special.gamma(nu_opt / 2)) / np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2))
    factor2 = 1 + ((dists - epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt-2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    log_Li = np.log(f_t)
    log_likelihood = np.nansum(log_Li)
    return -1 * log_likelihood

def minus_log_l_with_epsilon(d, dists, varis):
    sigma_opt = d[0]
    nu_opt = d[1]
    epsilon = d[2]

    coeff = (scipy.special.gamma((nu_opt + 1) / 2) / scipy.special.gamma(nu_opt / 2)) / np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2))
    factor2 = 1 + ((dists - epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt-2))
    f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
    log_Li = np.log(f_t)
    log_likelihood = np.nansum(log_Li)
    return -1 * log_likelihood

# Moved outside mle_t function
def run_opt(variant_adj, variant, nc_file_path, init_vals, bnds):
    # Open the NetCDF file within the process
    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        dists_here = open_nc_file['distances'][:,:,:,variant_adj].flatten()
        varis_here = open_nc_file['variances'][:,:,:,variant_adj].flatten()
    dists_here = dists_here[~np.isnan(dists_here)]
    varis_here = varis_here[~np.isnan(varis_here)]
    
    x_0 = init_vals
    if len(init_vals) > 2:
        res = minimize(minus_log_l_with_epsilon, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='L-BFGS-B')
    else:    
        res = minimize(minus_log_l, x_0, args=(dists_here, varis_here), bounds=[tuple(b) for b in bnds], method='L-BFGS-B')

    res_arr = [variant] + [res_var**2 if i == 0 else res_var for i, res_var in enumerate(res.x)]
    res_arr.append(-res.fun)

    res_arr = [str(el) for el in res_arr]
    
    return res_arr

"""
MPI functions
"""
def scatter(all_files):
    if CRANK == CROOT:
        # Split into CSIZE-1 chunks for workers only
        if len(all_files) < (CSIZE):
            # If fewer files than workers, assign one per worker and pad the rest with empty lists
            split_files = [list(chunk) for chunk in np.array_split(all_files, len(all_files))]
            split_files.extend([[] for _ in range((CSIZE) - len(split_files))])
        else:
            split_files = [list(chunk) for chunk in np.array_split(all_files, CSIZE)]

        # Root gets no work
        split_files = split_files
    else:
        split_files = None

    # Scatter so each rank gets its subset
    files_subset = COMM.scatter(split_files, root=CROOT)
    return files_subset

def worker(files_subset, path_to_data_files, save_here_dir, init_vals, bnds):

    # create results file if it doesn't exist
    rank_file = os.path.join(save_here_dir, f'mle_res_rank{CRANK}.csv')
    cols_here = get_mle_columns(init_vals)
    if os.path.exists(os.path.join(rank_file)):
        return None

    # create base csv for rank
    if not os.path.exists(rank_file):
        with open(rank_file, 'w') as f:
            f.write(','.join(cols_here) + '\n')

    # Each rank processes its subset of files
    for file in files_subset:
        nc_file_path = os.path.join(path_to_data_files, file)

        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data

        mle_arr = []
        for variant_ind, variant_raw in enumerate(variants):
            res = run_opt(variant_ind, variant_raw, nc_file_path, init_vals, bnds)
            mle_arr.append(res)

        mle_arr = sorted(mle_arr, key=lambda x: int(x[0]))

        # Save data
        with open(rank_file, 'a') as f:
            f.write('\n'.join([','.join(res) for res in mle_arr]) + '\n')

    return None

def get_num_of_variants(files_subset, path_to_data_files):
    num_variants = 0
    for file in files_subset:
        nc_file_path = os.path.join(path_to_data_files, file)
        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data
        num_variants += len(variants)
    return num_variants

def root(files_subset, path_to_data_files, save_here_dir, init_vals, bnds):

    num_variants = get_num_of_variants(files_subset, path_to_data_files)

    progress = tqdm(total=num_variants, desc="Processing root variants", unit="variant")
    # create results file if it doesn't exist
    rank_file = os.path.join(save_here_dir, f'mle_res_rank{CRANK}.csv')
    cols_here = get_mle_columns(init_vals)
    if os.path.exists(os.path.join(rank_file)):
        print('... Root MLE files already exists')
        return None

    # create base csv for rank
    if not os.path.exists(rank_file):
        with open(rank_file, 'w') as f:
            f.write(','.join(cols_here) + '\n')

    # Each rank processes its subset of files
    for file in files_subset:
        # check if file exists
        nc_file_path = os.path.join(path_to_data_files, file)

        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data

        mle_arr = []
        for variant_ind, variant_raw in enumerate(variants):
            res = run_opt(variant_ind, variant_raw, nc_file_path, init_vals, bnds)
            mle_arr.append(res)
            progress.update(1)

        mle_arr = sorted(mle_arr, key=lambda x: int(x[0]))

        # Save data
        with open(rank_file, 'a') as f:
            f.write('\n'.join([','.join(res) for res in mle_arr]) + '\n')
    progress.close()

    return None

def get_optimal_mle(all_mle_file, cols):

    optimal_vals = []
    with nc.Dataset(all_mle_file, 'r') as nc_file:

        log_likelihoods = nc_file['log_L'][:].data

        for col in cols:
            optimal_vals.append(nc_file[col][np.argmax(log_likelihoods)].data)
    
    return optimal_vals

"""
MLE with student-t distribution
"""

def mle_t(args):
    """
    MLE analysis using the student-t distribution approximation.
    """

    if CRANK == CROOT:
        # Load parameters from input file
        with open(args.input_file, 'r') as file:
            eval_params = json.load(file)
    else:
        eval_params = None
    eval_params = COMM.bcast(eval_params, root=CROOT)
    
    # Extract MLE optimization parameters
    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']
    cols_here = get_mle_columns(init_vals)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    path_to_data_files = save_here_dir + 'dists_varis_data/'

    if CRANK == CROOT:
        data_files = os.listdir(path_to_data_files)
        data_files = sorted(data_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    else:
        data_files = None
    files_subset = scatter(data_files)
    
    if CRANK == CROOT:
        print(f'Root processing: {files_subset}')

    if CRANK == CROOT:
        root(files_subset, path_to_data_files, save_here_dir, init_vals, bnds)
    else:
        worker(files_subset, path_to_data_files, save_here_dir, init_vals, bnds)
    
    # save to nc file
    COMM.Barrier()
    if CRANK == CROOT:
        save_mle_to_nc(save_here_dir, CSIZE, cols_here)

        all_mle_file = os.path.join(save_here_dir, 'all_mle.nc')
        optimal_vals = get_optimal_mle(all_mle_file, cols_here)
    else:
        return (None, None)

    return optimal_vals, cols_here