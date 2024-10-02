import numpy as np
import pandas as pd
import json
from .utils import calculate_distances_and_variances
from src.storage.utils import save_dataset, get_em_pred_filenames, save_distances_and_variances, create_distances_and_variances_base_files, get_variant_subsets
from .viz import plot_measurements


def model_discrepancy(args):
    """Collect datasets"""

    print('---------ModelDiscrepancy---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    satellite_file_path = eval_params['satellite_file_path']
    inputs_file_path = eval_params['emulator_inputs_file_path']
    subregion_filter = eval_params['subregion_filter']

    try:
        subset_size = eval_params['subset_size']
    except:
        subset_size = 5000

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path) ###
    num_variants = inputs_df.shape[0]

    # subregion work
    toggle_filter = subregion_filter['toggle_filter']
    if toggle_filter:
        lat_min = subregion_filter['lat_min']
        lat_max = subregion_filter['lat_max']
        lon_min = subregion_filter['lon_min']
        lon_max = subregion_filter['lon_max']

    # Import MODIS observations dataframe
    obs_df = pd.read_csv(satellite_file_path)
    obs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)
    snip_subregion_obs = obs_df[obs_df['time'] == np.unique(obs_df['time'])[0]]


    """
    If subregion of data is needed, show this subregion on plot
    """
    if toggle_filter:
        subregion_filt_idx_set = list((obs_df['latitude'] < lat_min) | (obs_df['longitude'] < lon_min) | (obs_df['latitude'] > lat_max) | (obs_df['longitude'] > lon_max))
        obs_df.loc[subregion_filt_idx_set , ['meanResponse', 'sdResponse']] = [float("nan"), float("nan")]

        plot_measurements(lon_min,
                          lat_min,
                          lon_max,
                          lat_max,
                          snip_subregion_obs,
                          save_here_dir)


    prediction_sets = get_em_pred_filenames(args)
    
    variant_subsets = get_variant_subsets(num_variants, subset_size)

    for subset in variant_subsets:
        create_distances_and_variances_base_files(save_here_dir, obs_df, subset)

    """
    Calculate distances and variances
    """
    calculate_distances_and_variances(args, num_variants, obs_df, prediction_sets, variant_subsets)

    return None
