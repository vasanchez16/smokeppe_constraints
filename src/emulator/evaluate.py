import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches 
import os
from datetime import datetime
import json
from .utils import get_em_pred_filenames, get_distances
from .viz import plot_measurements, plot_outliers
from tqdm import tqdm


def evaluate(args):

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    print(f'Run label: {run_label}')
    print('---------EmulatorEval---------')
    save_here_dir = args.output_dir + run_label + '/'

    emulator_folder_path = eval_params['emulator_output_folder_path']
    satellite_file_path = eval_params['satellite_file_path']
    inputs_file_path = eval_params['emulator_inputs_file_path']
    subregion_filter = eval_params['subregion_filter']

    # Extracting values from the subregion_filter dictionary
    toggle_filter = subregion_filter['toggle_filter']
    lat_min = subregion_filter['lat_min']
    lat_max = subregion_filter['lat_max']
    lon_min = subregion_filter['lon_min']
    lon_max = subregion_filter['lon_max']

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path)

    # Import MODIS observations dataframe
    my_obs_df = pd.read_csv(satellite_file_path)
    my_obs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)

    snip_subregion_obs = my_obs_df[my_obs_df['time'] == np.unique(my_obs_df['time'])[0]]
    if toggle_filter:
        plot_measurements(lon_min,
                          lat_min,
                          lon_max,
                          lat_max,
                          snip_subregion_obs,
                          save_here_dir)

    prediction_sets = get_em_pred_filenames(args)

    # Emulator Evaluation
    distances, variances, leastSqs = get_distances(inputs_df,
                                                   my_obs_df,
                                                   toggle_filter,
                                                   lat_min,
                                                   lon_min,
                                                   lat_max,
                                                   lon_max,
                                                   prediction_sets,
                                                   emulator_folder_path)

    #Here is where the sorting becomes very important as the distances and variances append to the incorrect corresponding gstp
    outliers_df = my_obs_df
    outliers_df['leastSquares'] = leastSqs
    outliers_df['distances'] = distances
    outliers_df['variances'] = variances

    #filter out the outliers
    plot_outliers(outliers_df, save_here_dir)
    
    # Create column to label missing data
    outliers_df['missing'] = np.isnan(outliers_df.leastSquares)
    # Create column to label points as outliers or above LS threshold
    outliers_df['outlier'] = False
    # Save new outliers csv with outlier and missing columns
    outliers_df.to_csv(save_here_dir + 'outliers.csv', index=True)
