import numpy as np
import pandas as pd
import json
from .utils import calculate_distances_and_variances
from src.storage.utils import save_dataset
from src.emulator.utils import get_em_pred_filenames


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

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path) ###
    num_variants = inputs_df.shape[0]

    # Import MODIS observations dataframe
    obs_df = pd.read_csv(satellite_file_path)

    prediction_sets = get_em_pred_filenames(args)

    """
    Calculate distances and variances
    """
    all_dists_df, all_vars_df = calculate_distances_and_variances(args, num_variants, obs_df, prediction_sets)


    """
    Save datasets
    """
    print('Saving distances.csv...')
    save_dataset(all_dists_df, save_here_dir + 'distances.csv')
    print('Saving variances.csv...')
    save_dataset(all_vars_df, save_here_dir + 'variances.csv')


    return all_dists_df, all_vars_df
