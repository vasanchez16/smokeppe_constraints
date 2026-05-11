import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import netCDF4 as nc
from src.storage.utils import save_distances_and_variances_one_time


def calculate_distances_and_variances(args, num_variants, obs_df, prediction_sets, variant_subsets):
    """
    Dispatch distance and variance calculations to the appropriate handler
    based on the file format of the emulator prediction sets.

    Arguments:
    - args: argparse.Namespace with input_file and output_dir attributes
    - num_variants: int, total number of emulator parameter variants
    - obs_df: pd.DataFrame, observation data (meanResponse, sdResponse, lat, lon, time)
    - prediction_sets: list of str, filenames for each emulator prediction file
    - variant_subsets: list of lists, groupings of variant indices for chunked NetCDF output
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    emulator_folder_path = eval_params['emulator_output_folder_path']

    my_obs_df = obs_df.copy()
    # Zero and NaN observations are treated as missing
    idxSet = (obs_df['meanResponse'] == 0) | (np.isnan(obs_df['meanResponse']))
    my_obs_df.loc[idxSet, ['meanResponse', 'sdResponse']] = [float('nan'), float('nan')]

    progress_bar = tqdm(total=len(prediction_sets), desc="Progress")

    if prediction_sets[0].endswith('.nc'):
        calcs_for_nc(my_obs_df, emulator_folder_path, prediction_sets, progress_bar, save_here_dir, variant_subsets)
    elif prediction_sets[0].endswith('.csv'):
        calcs_for_csv(my_obs_df, emulator_folder_path, prediction_sets, progress_bar, num_variants, save_here_dir, variant_subsets)


def calcs_for_nc(obs_df, emulator_folder_path, prediction_sets, progress_bar, save_here_dir, variant_subsets):
    """
    Calculate distances and total variances for all emulator variants from NetCDF files.

    For each time step, reads the emulator mean and standard deviation, then computes:
      distances  = observation - emulator mean        (shape: lat × lon × variant)
      variances  = obs variance + emulator variance   (shape: lat × lon × variant)

    Results are saved incrementally to NetCDF via save_distances_and_variances_one_time.

    Arguments:
    - obs_df: pd.DataFrame, observation data filtered to remove missing values
    - emulator_folder_path: str, directory containing emulator prediction NetCDF files
    - prediction_sets: list of str, prediction filenames, one per time step
    - progress_bar: tqdm object for tracking progress
    - save_here_dir: str, output directory for the current run
    - variant_subsets: list of lists, variant index groupings for chunked output files
    """
    for time_ind, (tm, prediction_set) in enumerate(zip(np.unique(obs_df.time), prediction_sets)):
        my_obs_df_this_time = obs_df[obs_df.time == tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude', 'longitude'], inplace=True, ignore_index=True)

        y_arr = my_obs_df_this_time['meanResponse'].values
        e_arr = my_obs_df_this_time['sdResponse'].values ** 2

        mean_res_arr, sd_res_arr = get_nc_data(emulator_folder_path, prediction_set)

        y_arr = np.reshape(y_arr, mean_res_arr.shape[:-1])
        e_arr = np.reshape(e_arr, mean_res_arr.shape[:-1])

        distances = y_arr[:, :, None] - mean_res_arr
        variances = e_arr[:, :, None] + sd_res_arr ** 2

        save_distances_and_variances_one_time(save_here_dir, distances, variances, tm, time_ind, variant_subsets)
        progress_bar.update(1)

    progress_bar.close()


def get_nc_data(emulator_folder_path, prediction_set):
    """
    Read emulator mean and standard deviation arrays from a NetCDF prediction file.

    Returns:
    - mean_res_arr: np.ndarray of shape (lat, lon, variant)
    - sd_res_arr:   np.ndarray of shape (lat, lon, variant)
    """
    with nc.Dataset(emulator_folder_path + prediction_set, 'r', format='NETCDF4') as nc_file:
        mean_res_arr = nc_file['meanResponse'][:, :, :]
        sd_res_arr = nc_file['sdResponse'][:, :, :]
    return mean_res_arr, sd_res_arr


def calcs_for_csv(obs_df, emulator_folder_path, prediction_sets, progress_bar, num_variants, save_here_dir, variant_subsets):
    """
    Calculate distances and total variances for all emulator variants from CSV files.

    Iterates over spatial pixels for each time step and computes:
      distances  = observation - emulator mean        (or NaN if observation is missing)
      variances  = obs variance + emulator variance   (or NaN if observation is missing)

    Results are saved incrementally to NetCDF via save_distances_and_variances_one_time.

    Arguments:
    - obs_df: pd.DataFrame, observation data filtered to remove missing values
    - emulator_folder_path: str, directory containing emulator prediction CSV files
    - prediction_sets: list of str, prediction filenames, one per time step
    - progress_bar: tqdm object for tracking progress
    - num_variants: int, number of emulator parameter variants
    - save_here_dir: str, output directory for the current run
    - variant_subsets: list of lists, variant index groupings for chunked output files
    """
    lats = obs_df['latitude'].unique()
    lons = obs_df['longitude'].unique()

    for time_ind, (tm, prediction_set) in enumerate(zip(np.unique(obs_df.time), prediction_sets)):
        my_obs_df_this_time = obs_df[obs_df.time == tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude', 'longitude'], inplace=True, ignore_index=True)

        mean_res_arr, sd_res_arr = get_csv_data(emulator_folder_path, prediction_set, obs_df, num_variants)

        obs_pixel = 0
        dists_lat_here_arr = []
        varis_lat_here_arr = []
        for lat_ind in range(len(lats)):
            dists_lon_here_arr = []
            varis_lon_here_arr = []
            for lon_ind in range(len(lons)):
                y = my_obs_df_this_time.loc[obs_pixel, 'meanResponse']
                e = my_obs_df_this_time.loc[obs_pixel, 'sdResponse'] ** 2

                zs = mean_res_arr[lat_ind, lon_ind, :]
                ss = sd_res_arr[lat_ind, lon_ind, :] ** 2

                if ~np.isnan(y) and y != 0:
                    distances = list(y - zs)
                    variances = list(e + ss)
                else:
                    distances = [float('nan')] * len(zs)
                    variances = [float('nan')] * len(zs)
                obs_pixel += 1

                dists_lon_here_arr.append(distances)
                varis_lon_here_arr.append(variances)

            dists_lat_here_arr.append(dists_lon_here_arr)
            varis_lat_here_arr.append(varis_lon_here_arr)

        save_distances_and_variances_one_time(save_here_dir, dists_lat_here_arr, varis_lat_here_arr, tm, time_ind, variant_subsets)
        progress_bar.update(1)

    progress_bar.close()


def get_csv_data(emulator_folder_path, prediction_set, obs_df, num_variants):
    """
    Read and reshape emulator predictions from a CSV file.

    Sorts by latitude, longitude, and variant, then reshapes the flat arrays into
    (lat, lon, variant) grids using the unique spatial coordinates in obs_df.

    Returns:
    - mean_res_arr: np.ndarray of shape (lat, lon, variant)
    - sd_res_arr:   np.ndarray of shape (lat, lon, variant)
    """
    prediction_data = pd.read_csv(emulator_folder_path + prediction_set)
    prediction_data.sort_values(['latitude', 'longitude', 'variant'], inplace=True, ignore_index=True)

    n_lats = len(obs_df['latitude'].unique())
    n_lons = len(obs_df['longitude'].unique())

    mean_res_arr = np.reshape(prediction_data['meanResponse'].values, (n_lats, n_lons, num_variants))
    sd_res_arr = np.reshape(prediction_data['sdResponse'].values, (n_lats, n_lons, num_variants))

    return mean_res_arr, sd_res_arr
