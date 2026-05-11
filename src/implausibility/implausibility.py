import numpy as np
import pandas as pd
import os
import json
import netCDF4 as nc
import multiprocessing as mp
from .utils import calculate_implausibility


def implausibilities(args):
    """
    Compute the implausibility statistic I(u^k) for every emulator variant.

    Reads the MLE results and pre-computed distances/variances, runs
    calculate_implausibility in parallel across data files, and writes
    implausibilities.csv. Also saves the distances and variances for the
    most plausible and max-likelihood variants for downstream diagnostics.
    """
    print('---------Implausibilities---------')

    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']
    path_to_data_files = save_here_dir + 'dists_varis_data/'

    inputs_df = pd.read_csv(eval_params['emulator_inputs_file_path'])
    num_variants = inputs_df.shape[0]

    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    mle_variant = int(mle_df['parameter_set_num'].values[0])
    additional_variance = float(mle_df['variance_mle'].values[0])

    try:
        epsilon = float(mle_df['epsilon'].values[0])
    except KeyError:
        epsilon = 0

    nu = float(mle_df['nu'].values[0]) if 'student-t' in stats_dist_method else 0

    with open(save_here_dir + 'implausibilities.csv', 'w') as implaus_file:
        implaus_file.write('variant,I\n')

    data_files = sorted(
        os.listdir(path_to_data_files),
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )

    num_processes = min(mp.cpu_count(), len(data_files))

    with mp.Pool(processes=num_processes) as pool:
        futures = [
            pool.apply_async(calculate_implausibility,
                             args=(path_to_data_files + file, stats_dist_method, additional_variance, nu, epsilon))
            for file in data_files
        ]
        many_implaus_arrs = [f.get() for f in futures]

    min_implaus_arr = None
    for chunk, file in zip(many_implaus_arrs, data_files):
        print(f'Appending Implaus for {file}...')
        chunk_str = '\n'.join(','.join(i) for i in chunk) + '\n'

        best_variant_data_here = sorted(chunk, key=lambda x: float(x[-1]))[0]

        if min_implaus_arr is None or best_variant_data_here[-1] < min_implaus_arr[-1]:
            min_implaus_arr = best_variant_data_here

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

    def _extract_dists_varis(data_file_ind, variant_num):
        """Read and flatten distances/variances for a single variant from its data file."""
        with nc.Dataset(path_to_data_files + data_files[data_file_ind], 'r') as nc_file:
            variants = nc_file['variant'][:].data
            idx = variant_num - min(variants)
            dists = nc_file['distances'][:, :, :, idx].flatten()
            varis = nc_file['variances'][:, :, :, idx].flatten()
        return pd.DataFrame({'dists': dists, 'varis': varis})

    _extract_dists_varis(best_param_data_file_ind, best_param_set_num).to_csv(
        save_here_dir + 'mostPlausibleDistsVaris.csv', index=False
    )
    _extract_dists_varis(mle_data_file_ind, mle_variant).to_csv(
        save_here_dir + 'maxLikelihoodDistsVaris.csv', index=False
    )
