import numpy as np
import netCDF4 as nc
from math import gamma
from scipy.optimize import minimize
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


# --- Optimization functions ---

def minus_log_l(d, dists, varis):
    """Negative log-likelihood for the student-t distribution without a bias term."""
    try:
        sigma_opt = d[0]
        nu_opt = d[1]

        coeff = (gamma((nu_opt + 1) / 2) / gamma(nu_opt / 2)) / \
                np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2))
        factor2 = 1 + (dists**2) / ((varis + sigma_opt**2) * (nu_opt - 2))

        if np.any(factor2 <= 0) or np.any(~np.isfinite(coeff)):
            return np.inf

        f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
        log_Li = np.log(f_t)

        if np.any(~np.isfinite(log_Li)):
            return np.inf

        return -np.sum(log_Li)

    except Exception:
        return np.inf


def minus_log_l_with_epsilon(d, dists, varis):
    """Negative log-likelihood for the student-t distribution with a bias term (epsilon)."""
    try:
        sigma_opt = d[0]
        nu_opt = d[1]
        epsilon = d[2]

        coeff = (gamma((nu_opt + 1) / 2) / gamma(nu_opt / 2)) / \
                np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2))
        factor2 = 1 + ((dists - epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt - 2))

        if np.any(factor2 <= 0) or np.any(~np.isfinite(coeff)):
            return np.inf

        f_t = coeff * factor2 ** (-1 * (nu_opt + 1) / 2)
        log_Li = np.log(f_t)

        if np.any(~np.isfinite(log_Li)):
            return np.inf

        return -np.sum(log_Li)

    except Exception:
        return np.inf


def run_opt(variant_adj, variant, nc_file_path, init_vals, bnds):
    """
    Run bounded MLE optimization for a single emulator variant.

    Reads distances and variances from the NetCDF file, drops NaNs, then
    minimizes the negative log-likelihood with L-BFGS-B. The first decision
    variable (sigma) is squared in the result to produce variance_mle.
    """
    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        dists_here = open_nc_file['distances'][:, :, :, variant_adj].flatten()
        varis_here = open_nc_file['variances'][:, :, :, variant_adj].flatten()
    dists_here = dists_here[~np.isnan(dists_here)]
    varis_here = varis_here[~np.isnan(varis_here)]

    x_0 = init_vals
    if len(init_vals) > 2:
        res = minimize(minus_log_l_with_epsilon, x_0, args=(dists_here, varis_here),
                       bounds=[tuple(b) for b in bnds], method='L-BFGS-B')
    else:
        res = minimize(minus_log_l, x_0, args=(dists_here, varis_here),
                       bounds=[tuple(b) for b in bnds], method='L-BFGS-B')

    res_arr = [variant] + [val**2 if i == 0 else val for i, val in enumerate(res.x)]
    res_arr.append(-res.fun)
    return [str(el) for el in res_arr]


# --- MPI functions ---

def scatter(all_files):
    """
    Scatter the list of data files across all MPI ranks.

    If there are fewer files than ranks, pads with empty lists so every rank
    receives a (possibly empty) subset.
    """
    if CRANK == CROOT:
        if len(all_files) < CSIZE:
            split_files = [list(chunk) for chunk in np.array_split(all_files, len(all_files))]
            split_files.extend([[] for _ in range(CSIZE - len(split_files))])
        else:
            split_files = [list(chunk) for chunk in np.array_split(all_files, CSIZE)]
    else:
        split_files = None

    return COMM.scatter(split_files, root=CROOT)


def worker(files_subset, path_to_data_files, save_here_dir, init_vals, bnds):
    """Process the assigned file subset and write MLE results to a per-rank CSV."""
    rank_file = os.path.join(save_here_dir, f'mle_res_rank{CRANK}.csv')
    cols_here = get_mle_columns(init_vals)

    if not os.path.exists(rank_file):
        with open(rank_file, 'w') as f:
            f.write(','.join(cols_here) + '\n')

    for file in files_subset:
        nc_file_path = os.path.join(path_to_data_files, file)

        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data

        mle_arr = [run_opt(i, v, nc_file_path, init_vals, bnds) for i, v in enumerate(variants)]
        mle_arr = sorted(mle_arr, key=lambda x: int(x[0]))

        with open(rank_file, 'a') as f:
            f.write('\n'.join(','.join(res) for res in mle_arr) + '\n')


def get_num_of_variants(files_subset, path_to_data_files):
    """Return the total number of emulator variants across all files in the subset."""
    num_variants = 0
    for file in files_subset:
        nc_file_path = os.path.join(path_to_data_files, file)
        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data
        num_variants += len(variants)
    return num_variants


def root(files_subset, path_to_data_files, save_here_dir, init_vals, bnds):
    """Process the root rank's file subset with a tqdm progress bar."""
    num_variants = get_num_of_variants(files_subset, path_to_data_files)
    progress = tqdm(total=num_variants, desc="Processing root variants", unit="variant")

    rank_file = os.path.join(save_here_dir, f'mle_res_rank{CRANK}.csv')
    cols_here = get_mle_columns(init_vals)

    if not os.path.exists(rank_file):
        with open(rank_file, 'w') as f:
            f.write(','.join(cols_here) + '\n')

    for file in files_subset:
        nc_file_path = os.path.join(path_to_data_files, file)

        with nc.Dataset(nc_file_path, 'r') as nc_file:
            variants = nc_file['variant'][:].data

        mle_arr = []
        for variant_ind, variant_raw in enumerate(variants):
            res = run_opt(variant_ind, variant_raw, nc_file_path, init_vals, bnds)
            mle_arr.append(res)
            progress.update(1)

        mle_arr = sorted(mle_arr, key=lambda x: int(x[0]))

        with open(rank_file, 'a') as f:
            f.write('\n'.join(','.join(res) for res in mle_arr) + '\n')

    progress.close()


def get_optimal_mle(all_mle_file, cols):
    """Read the merged all_mle NetCDF file and return values for the max-likelihood variant."""
    with nc.Dataset(all_mle_file, 'r') as nc_file:
        log_likelihoods = nc_file['log_L'][:].data
        optimal_vals = [nc_file[col][np.argmax(log_likelihoods)].data for col in cols]
    return optimal_vals


# --- MLE with student-t distribution ---

def mle_t(args):
    """
    Run MLE analysis using the student-t distribution approximation with MPI parallelism.

    The root rank reads and broadcasts the config, then scatters data file subsets
    to all ranks. Each rank writes its optimization results to a per-rank CSV.
    After all ranks finish (barrier), the root merges results into a NetCDF file
    and returns the optimized values for the maximum-likelihood variant.
    """
    if CRANK == CROOT:
        with open(args.input_file, 'r') as file:
            eval_params = json.load(file)
    else:
        eval_params = None
    eval_params = COMM.bcast(eval_params, root=CROOT)

    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']
    cols_here = get_mle_columns(init_vals)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    path_to_data_files = save_here_dir + 'dists_varis_data/'

    if CRANK == CROOT:
        data_files = sorted(
            os.listdir(path_to_data_files),
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )
    else:
        data_files = None
    files_subset = scatter(data_files)

    if CRANK == CROOT:
        root(files_subset, path_to_data_files, save_here_dir, init_vals, bnds)
    else:
        worker(files_subset, path_to_data_files, save_here_dir, init_vals, bnds)

    COMM.Barrier()
    if CRANK == CROOT:
        save_mle_to_nc(save_here_dir, CSIZE, cols_here)
        all_mle_file = os.path.join(save_here_dir, 'all_mle.nc')
        optimal_vals = get_optimal_mle(all_mle_file, cols_here)
    else:
        return (None, None)

    return optimal_vals, cols_here
