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

    if '.nc' in prediction_sets[0]:
        all_dists_arr, all_varis_arr = calcs_for_nc(my_obs_df, emulator_folder_path, prediction_sets, progress_bar)

        return all_dists_arr, all_varis_arr
    
    # if '.csv' in prediction_set[0]:
    #     all_dists_arr, all_varis_arr = calcs_for_csv(my_obs_df, emulator_folder_path, prediction_sets, progress_bar)
        
    #     return all_dists_arr, all_varis_arr

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

def calcs_for_nc(obs_df, emulator_folder_path, prediction_sets, progress_bar):
    allDistances = []
    allVariances = []

    lats = obs_df['latitude'].unqiue()
    lons = obs_df['longitude'].unique()

    for tm, prediction_set in zip(np.unique(obs_df.time), prediction_sets):
        # time is a datetime string in this case, but df here has time in hours as float
        my_obs_df_this_time = obs_df[obs_df.time==tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude','longitude'], inplace=True, ignore_index=True)

        mean_res_arr, sd_res_arr = get_nc_data(emulator_folder_path, prediction_set) # dims: lat, lon, variant

        obs_pixel = 0
        for lat_ind in range(len(lats)):
            for lon_ind in range(len(lons)):
                y = my_obs_df_this_time.loc[obs_pixel, 'meanResponse']
                e = my_obs_df_this_time.loc[obs_pixel, 'sdResponse']**2

                zs = mean_res_arr[lat_ind,lon_ind,:]
                ss = sd_res_arr[lat_ind,lon_ind,:]**2

                if ~np.isnan(y) and y != 0:
                    distances = list(y - zs)
                    variances = list(e + ss)
                else:
                    distances = [float('nan')]*len(zs)
                    variances = [float('nan')]*len(zs)
                allDistances.append(distances)
                allVariances.append(variances)
                obs_pixel += 1

        # print(f'Done with {prediction_set}')
        progress_bar.update(1)
    progress_bar.close()
    return allDistances, allVariances

def get_nc_data(emulator_folder_path, prediction_set):
    nc_file = nc.Dataset(emulator_folder_path + prediction_set, 'r', format='NETCDF4')

    mean_res_arr = nc_file['meanResponse']
    sd_res_arr = nc_file['sdResponse']

    return mean_res_arr, sd_res_arr

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