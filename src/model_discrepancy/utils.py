import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import netCDF4 as nc


def calculate_distances_and_variances(args, num_variants, obs_df, prediction_sets):
    """
    Calculate distances and variances based on the given inputs and observations.

    Arguments:
    - inputs_df: DataFrame containing input data
    - obs_df: DataFrame containing observation data
    - idxSet: Set of row indices to exclude from analysis
    - ocean_smokeppe_dir: Directory path for ocean smokeppe data
    - prediction_sets: List of prediction sets

    Returns:
    - all_dists_df: DataFrame containing distances
    - all_vars_df: DataFrame containing variances
    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    emulator_folder_path = eval_params['emulator_output_folder_path']

    allDistances = []
    allVariances = []

    my_obs_df = obs_df.copy()
    idxSet = (obs_df['meanResponse'] == 0) | (np.isnan(obs_df['meanResponse']))
    # set missing
    my_obs_df.loc[idxSet, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]
    progress_bar = tqdm(total=len(prediction_sets), desc="Progress")

    for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        # time is a datetime string in this case, but df here has time in hours as float
        my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude','longitude'], inplace=True, ignore_index=True)
        num_pixels = len(my_obs_df_this_time.index)

        my_predict_df_this_time = read_in_prediction_data(emulator_folder_path, prediction_set)
        my_predict_df_this_time.sort_values(['latitude','longitude','variant'], inplace=True, ignore_index=True)
        if 'meanResponse' not in my_predict_df_this_time.columns and 'mean' in my_predict_df_this_time.columns:
            my_predict_df_this_time.rename(columns={'mean':'meanResponse'}, inplace=True)
        if 'sdResponse' not in my_predict_df_this_time.columns and 'std' in my_predict_df_this_time.columns:
            my_predict_df_this_time.rename(columns={'std':'sdResponse'}, inplace=True)
        # opens csv data that stores emulated data for each point, csv's are labeled by time
        # print(f'Read in {prediction_set}')

        my_predict_dfs = [
            my_predict_df_this_time.iloc[k*num_variants:(k+1)*num_variants, :].reset_index(drop=True)
            for k in range(num_pixels)
        ]

        for row in range(num_pixels):
            y = my_obs_df_this_time.loc[row, 'meanResponse']
            e = my_obs_df_this_time.loc[row, 'sdResponse']**2

            zs = my_predict_dfs[row]['meanResponse']
            ss = my_predict_dfs[row]['sdResponse']**2

            if ~np.isnan(y) and y != 0:
                distances = list(y - zs)
                variances = list(e + ss)
            else:
                distances = [float('nan')]*len(zs)
                variances = [float('nan')]*len(zs)

            allDistances.append(pd.DataFrame(distances).transpose())
            allVariances.append(pd.DataFrame(variances).transpose())
        # print(f'Done with {prediction_set}')
        progress_bar.update(1)
    progress_bar.close()

    print('Concatenating distances...')
    all_dists_df = pd.concat(allDistances, axis=0).reset_index(drop=True)
    print('Concatenating variances...')
    all_vars_df = pd.concat(allVariances, axis=0).reset_index(drop=True)

    return all_dists_df, all_vars_df

def read_in_prediction_data(emulator_folder_path, prediction_set):
    if '.nc' in prediction_set:
        data = read_nc_pred_file(emulator_folder_path, prediction_set)
    elif '.csv' in prediction_set:
        data = pd.read_csv(emulator_folder_path + prediction_set)
    return data

def read_nc_pred_file(emulator_folder_path, prediction_set):
    nc_file = nc.Dataset(emulator_folder_path + prediction_set, 'r', format='NETCDF4')

    data = pd.DataFrame()
    lats = [lat for lat in nc_file['latitude'][:].data for lon in nc_file['longitude'][:].data]
    data['latitude'] = np.repeat(lats,len(nc_file['variant'][:].data))
    lons = [lon for lat in nc_file['latitude'][:].data for lon in nc_file['longitude'][:].data]
    data['longitude'] = np.repeat(lons,len(nc_file['variant'][:].data))

    data['meanResponse'] = nc_file['meanResponse'][:].data.flatten()
    data['sdResponse'] = nc_file['sdResponse'][:].data.flatten()

    num_points = int(len(nc_file['meanResponse'][:].flatten()) / len(nc_file['variant'][:].data))
    data['variant'] = np.tile(nc_file['variant'][:].data, num_points)

    return data