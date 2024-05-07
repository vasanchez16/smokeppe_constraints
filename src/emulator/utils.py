import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_em_pred_filenames(args):
    """
    getting sorted list of the em prediciton filenames
    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    emulator_folder_path = eval_params['emulator_output_folder_path']

    folder_contents = os.listdir(emulator_folder_path)
    folder_contents.sort()

    return folder_contents


def get_distances(inputs_df, my_obs_df, toggle_filter, lat_min, lon_min, lat_max, lon_max, obsSdCensor, prediction_sets, ls_thresh, save_here_dir, emulator_folder_path):
    # making empty data storage lists for last calculations
    which_gets_least_squares = []
    distances = []
    variances = []

    num_variants = inputs_df.shape[0]

    if toggle_filter:
        subregion_filt_idx_set = list((my_obs_df['latitude'] < lat_min) | (my_obs_df['longitude'] < lon_min) | (my_obs_df['latitude'] > lat_max) | (my_obs_df['longitude'] > lon_max))
        my_obs_df.loc[subregion_filt_idx_set , ['meanResponse', 'sdResponse']] = [float("nan"), float("nan")]

    # my_obs_df.loc[my_obs_df.sdResponse >= obsSdCensor, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]
    progress_bar = tqdm(total=len(prediction_sets), desc="Progress")
    for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        # print(tm,prediction_set)
        my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index) # give the number of lat_long points for one time

        #need to make df of prediction outputs for each day
        my_predict_df_this_time = pd.read_csv(emulator_folder_path + prediction_set) ###
        my_predict_df_this_time.sort_values(['latitude','longitude','variant'],inplace=True, ignore_index=True)
        if 'meanResponse' not in my_predict_df_this_time.columns and 'mean' in my_predict_df_this_time.columns:
            my_predict_df_this_time.rename(columns={'mean':'meanResponse'}, inplace=True)
        if 'sdResponse' not in my_predict_df_this_time.columns and 'std' in my_predict_df_this_time.columns:
            my_predict_df_this_time.rename(columns={'std':'sdResponse'}, inplace=True)

        #makes a list of df's, each df represents a different gstp
        my_predict_dfs = [
            my_predict_df_this_time.iloc[k*num_variants:(k+1)*num_variants, :].reset_index(drop=True) 
            for k in range(num_pixels)
        ]

        # Check which row (test variant) gives least squares
        for row in range(num_pixels):
            # each row in obs_df is a different lat_long point
            y = my_obs_df_this_time.loc[row, 'meanResponse'] 
            e = my_obs_df_this_time.loc[row, 'sdResponse']**2 
            # each element in pred df's list represents a dif lat_long point
            zs = my_predict_dfs[row]['meanResponse']
            ss = my_predict_dfs[row]['sdResponse']**2

            if ~np.isnan(y) and y != 0:
                squares = list((y - zs)**2 / (e + ss))
                # least_squares = np.percentile(squares,10, method='closest_observation')
                least_squares = min(squares) #takes best value from squares list
                idx = squares.index(least_squares) #finds the variant number of this best value
                
                distances.append(y-zs[idx])
                variances.append(e + ss[idx])
            else:
                distances.append(float("nan"))
                variances.append(float("nan"))
        # print(f'{prediction_set} completed.')
        progress_bar.update(1)
    progress_bar.close()
    leastSqs = [np.abs(distances[k]) / np.sqrt(variances[k]) for k in range(len(distances))]
    return distances, variances, leastSqs
