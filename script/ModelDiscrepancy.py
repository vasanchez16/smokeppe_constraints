import numpy as np
import pandas as pd
import json
from src.inference import calculate_distances_and_variances
from src.inference.utils import save_dataset, get_em_pred_filenames

def ModelDiscrepancy(args):
    """Collect datasets"""

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    emulator_folder_path = eval_params['emulator_output_folder_path']
    satellite_file_path = eval_params['satellite_file_path']
    inputs_file_path = eval_params['emulator_inputs_file_path']

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path,index_col=0) ###
    num_variants = inputs_df.shape[0]

    # Import MODIS observations dataframe
    obs_df = pd.read_csv(save_here_dir + 'outliers.csv', index_col=0)

    prediction_sets = get_em_pred_filenames(args)
    prediction_sets = prediction_sets[:28] # this line is temporary

    """
    Calculate distances and variances
    """
    all_dists_df, all_vars_df = calculate_distances_and_variances(args, num_variants, obs_df, args.output_dir, prediction_sets)


    """
    Save datasets
    """
    save_dataset(all_dists_df, save_here_dir + 'distances.csv')
    save_dataset(all_vars_df, save_here_dir + 'variances.csv')


    return
