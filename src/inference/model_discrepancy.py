import numpy as np
import pandas as pd
import json


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
    idxSet=list((obs_df['missing']) | (obs_df['outlier'])) ###
    # set missing or outlier values to nan
    my_obs_df.loc[idxSet, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]

    for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        # time is a datetime string in this case, but df here has time in hours as float
        my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index)

        my_predict_df_this_time = pd.read_csv(emulator_folder_path + prediction_set , index_col=0)
        my_predict_df_this_time.sort_values(['latitude','longitude','variant'], inplace=True, ignore_index=True)
        # opens csv data that stores emulated data for each point, csv's are labeled by time
        print(f'Read in {prediction_set}')

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
        print(f'Done with {prediction_set}')

    print('Concatenating dists...')
    all_dists_df = pd.concat(allDistances, axis=0).reset_index(drop=True)
    print('Concatenating Varis...')
    all_vars_df = pd.concat(allVariances, axis=0).reset_index(drop=True)
    print('saving dists...')

    return all_dists_df, all_vars_df
