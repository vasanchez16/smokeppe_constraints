import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import netCDF4 as nc
from src.storage.utils import save_distances_and_variances_one_time


def calculate_distances_and_variances(args, num_variants, obs_df, prediction_sets, variant_subsets):
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

    my_obs_df = obs_df.copy()
    idxSet = (obs_df['meanResponse'] == 0) | (np.isnan(obs_df['meanResponse']))
    # set missing
    my_obs_df.loc[idxSet, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]
    progress_bar = tqdm(total=len(prediction_sets), desc="Progress")

    # run this calc method for nc files
    if '.nc' in prediction_sets[0]:
        calcs_for_nc(my_obs_df, emulator_folder_path, prediction_sets, progress_bar, save_here_dir, variant_subsets)

        return None
    
    if '.csv' in prediction_sets[0]:
        calcs_for_csv(my_obs_df, emulator_folder_path, prediction_sets, progress_bar, num_variants, save_here_dir, variant_subsets)

        return None

    return None

def calcs_for_nc(obs_df, emulator_folder_path, prediction_sets, progress_bar, save_here_dir, variant_subsets):
    """
    Function used to calculate the distances and total variances for all emulator variants. Specifically dedicated to netCDF files.

    Arguments:
    obs_df: pd.DataFrame
    DataFrame containing the measurement data.
    emulator_folder_path: str
    Path to the folder containing emulator predicition data.
    prediction_sets:[str]
    File names for all the emulator prediction files.
    progress_bar: tqdm obj
    Used to keep track of data calculation progress.

    Returns:
    allDistances: numpy array
    Array containing all the distances data (Measurement - Emulator) for all emulator variants.
    Dimensions: (time, latitude, longitude, variant).
    allVarainces: numpy array
    Array containing all the variances data (Measurement variance + Emulator Variance) for all emulator variants.
    Dimensions: (time, latitude, longitude, variant).
    """

    # dimension of spatial coverage
    lats = obs_df['latitude'].unique()
    lons = obs_df['longitude'].unique()

    time_ind = 0
    for tm, prediction_set in zip(np.unique(obs_df.time), prediction_sets):
        # pick subset of observation data and sort
        my_obs_df_this_time = obs_df[obs_df.time==tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude','longitude'], inplace=True, ignore_index=True)

        # get predictions and prediction uncertainties
        mean_res_arr, sd_res_arr = get_nc_data(emulator_folder_path, prediction_set) # dims: lat, lon, variant

        # calc distances and total variances
        obs_pixel = 0
        # store data for one lat here
        dists_lat_here_arr = []
        varis_lat_here_arr = []
        for lat_ind in range(len(lats)):
            # store data for one lon here
            dists_lon_here_arr = []
            varis_lon_here_arr = []
            for lon_ind in range(len(lons)):
                #get observation data
                y = my_obs_df_this_time.loc[obs_pixel, 'meanResponse']
                e = my_obs_df_this_time.loc[obs_pixel, 'sdResponse']**2

                # get emulator data
                zs = mean_res_arr[lat_ind,lon_ind,:]
                ss = sd_res_arr[lat_ind,lon_ind,:]**2

                if ~np.isnan(y) and y != 0 and e != 0:
                    # find observation - emulator difference
                    distances = list(y - zs)
                    #  find total variance from measurement and emulator
                    variances = list(e + ss)
                else:
                    # set dist and varis equal to nan if obs is missing or zero
                    distances = [float('nan')]*len(zs)
                    variances = [float('nan')]*len(zs)
                obs_pixel += 1

                dists_lon_here_arr.append(distances)
                varis_lon_here_arr.append(variances)

            dists_lat_here_arr.append(dists_lon_here_arr)
            varis_lat_here_arr.append(varis_lon_here_arr)

        # saves dists and varis for one time output to existing nc file
        save_distances_and_variances_one_time(save_here_dir, dists_lat_here_arr, varis_lat_here_arr, tm, time_ind, variant_subsets)

        # update progress bar
        time_ind += 1
        progress_bar.update(1)
    
    # close progress bar
    progress_bar.close()
    return None

def get_nc_data(emulator_folder_path, prediction_set):
    nc_file = nc.Dataset(emulator_folder_path + prediction_set, 'r', format='NETCDF4')

    # save prediction and prediction uncertainty
    mean_res_arr = nc_file['meanResponse'][:,:,:]
    sd_res_arr = nc_file['sdResponse'][:,:,:]

    nc_file.close()

    return mean_res_arr, sd_res_arr

def calcs_for_csv(obs_df, emulator_folder_path, prediction_sets, progress_bar, num_variants, save_here_dir, variant_subsets):
    """
    Function used to calculate the distances and total variances for all emulator variants. Specifically dedicated to csv files.
    """
    # dimension of spatial coverage
    lats = obs_df['latitude'].unique()
    lons = obs_df['longitude'].unique()
    
    time_ind = 0
    for tm, prediction_set in zip(np.unique(obs_df.time), prediction_sets):
        # pick subset of observation data and sort
        my_obs_df_this_time = obs_df[obs_df.time==tm].reset_index(drop=True)
        my_obs_df_this_time.sort_values(['latitude','longitude'], inplace=True, ignore_index=True)

        # get predictions and prediction uncertainties
        mean_res_arr, sd_res_arr = get_csv_data(emulator_folder_path, prediction_set, obs_df, num_variants) # dims: lat, lon, variant

        # calc distances and total variances
        obs_pixel = 0
        # store data for one lat here
        dists_lat_here_arr = []
        varis_lat_here_arr = []
        for lat_ind in range(len(lats)):
            # store data for one lon here
            dists_lon_here_arr = []
            varis_lon_here_arr = []
            for lon_ind in range(len(lons)):
                #get observation data
                y = my_obs_df_this_time.loc[obs_pixel, 'meanResponse']
                e = my_obs_df_this_time.loc[obs_pixel, 'sdResponse']**2

                # get emulator data
                zs = mean_res_arr[lat_ind,lon_ind,:]
                ss = sd_res_arr[lat_ind,lon_ind,:]**2

                if ~np.isnan(y) and y != 0:
                    # find observation - emulator difference
                    distances = list(y - zs)
                    #  find total variance from measurement and emulator
                    variances = list(e + ss)
                else:
                    # set dist and varis equal to nan if obs is missing or zero
                    distances = [float('nan')]*len(zs)
                    variances = [float('nan')]*len(zs)
                obs_pixel += 1

                dists_lon_here_arr.append(distances)
                varis_lon_here_arr.append(variances)

            dists_lat_here_arr.append(dists_lon_here_arr)
            varis_lat_here_arr.append(varis_lon_here_arr)

        # saves dists and varis for one time output to existing nc file
        save_distances_and_variances_one_time(save_here_dir, dists_lat_here_arr, varis_lat_here_arr, tm, time_ind, variant_subsets)

        # update progress bar
        time_ind += 1
        progress_bar.update(1)
    
    # close progress bar
    progress_bar.close()
    return None

def get_csv_data(emulator_folder_path, prediction_set, obs_df, num_variants):

    prediction_data = pd.read_csv(emulator_folder_path + prediction_set)
    prediction_data.sort_values(['latitude','longitude','variant'], inplace=True, ignore_index=True)

    mean_data = prediction_data['meanResponse'].values
    sd_data = prediction_data['sdResponse'].values

    mean_res_arr = np.reshape(mean_data,(len(obs_df['latitude'].unique()), len(obs_df['longitude'].unique()), num_variants))
    sd_res_arr = np.reshape(sd_data,(len(obs_df['latitude'].unique()), len(obs_df['longitude'].unique()), num_variants))

    return mean_res_arr, sd_res_arr