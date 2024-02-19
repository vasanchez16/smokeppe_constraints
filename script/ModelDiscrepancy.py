import numpy as np
import pandas as pd
from src.inference import calculate_distances_and_variances
from src.utils import save_dataset, save_indexed_dataset

def model_discrepancy(args):
    """Collect datasets"""
    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(ocean_smokeppe_dir + 'emulatorVariants10k.csv',index_col=0) ###

    # Import MODIS observations dataframe
    obs_df = pd.read_csv(data_folder + 'outliers.csv', index_col=0)

    # Making filenames to read predictions csvs
    days = [str(n).zfill(2) for n in range(1, 32)]
    times = ["09_20_00", "12_20_00"]

    # Since the predictions take up so much space, they are separated by day
    prediction_sets_aug = ["predictions_08_" + day + "_17_" + time for day in days for time in times] ###

    days = [str(n).zfill(2) for n in range(1, 31)]
    times = ["09_20_00", "12_20_00"]

    # Since the predictions take up so much space, they are separated by day
    prediction_sets_sept = ["predictions_09_" + day + "_17_" + time for day in days for time in times] ###

    prediction_sets = prediction_sets_aug + prediction_sets_sept
    prediction_sets = prediction_sets[:28] # this line is temporary

    idxSet=list((obs_df['missing']) | (obs_df['outlier'])) ###

    """
    Calculate distances and variances
    """
    calculate_distances_and_variances(inputs_df, obs_df, idxSet, ocean_smokeppe_dir, prediction_sets)


    """
    Save datasets
    """
    save_dataset(all_dists_df, data_folder + 'distances.csv')
    save_dataset(all_vars_df, data_folder + 'variances.csv')

    if args.save_best:
        save_indexed_dataset(all_dists_df, 8210, data_folder + 'dists_8210.csv')
        save_indexed_dataset(all_vars_df, 8210, data_folder + 'vars_8210.csv')

    return
