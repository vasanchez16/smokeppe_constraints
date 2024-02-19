import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as patches 
import os
from datetime import datetime
import json
import click
@click.command()
@click.argument('eval_parameters_file_path', type=str)

def runEmEval(eval_parameters_file_path):
    ###############################################################################
    with open(eval_parameters_file_path,'r') as file:
        evaluation_parameters = json.load(file)

    save_here_dir = evaluation_parameters['save_here_dir']
    run_label = evaluation_parameters['run_label']
    emulator_folder_path = evaluation_parameters['emulator_folder_path']
    satellite_file_path = evaluation_parameters['satellite_file_path']
    obsSdCensor = evaluation_parameters['obsSdCensor']
    inputs_file_path = evaluation_parameters['inputs_file_path']
    subregion_filter = evaluation_parameters['subregion_filter']

    # Extracting values from the subregion_filter dictionary
    toggle_filter = subregion_filter['toggle_filter']
    lat_min = subregion_filter['lat_min']
    lat_max = subregion_filter['lat_max']
    lon_min = subregion_filter['lon_min']
    lon_max = subregion_filter['lon_max']


    if not os.path.exists(save_here_dir + run_label):
        os.mkdir(save_here_dir + run_label)

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path,index_col=0)

    # Import MODIS observations dataframe
    my_obs_df = pd.read_csv(satellite_file_path,index_col=0)
    my_obs_df.sort_values(['time','longitude','latitude'], inplace=True, ignore_index=True)

    if toggle_filter:
        plt.figure(figsize=(12,10))
        ax = plt.subplot(111,projection=ccrs.PlateCarree())
        ax.set_extent([-60,45,-45,15], crs=ccrs.PlateCarree())
        plt.gca().coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

        rect1 = patches.Rectangle((lon_low, lat_low), 
                                lon_high - lon_low, 
                                lat_high - lat_low, 
                                linestyle='--', edgecolor='red', facecolor='none', linewidth=4, label='Subregion'
                                )
        ax.add_patch(rect1)
        gl.xlabel_style = {'fontsize':20}
        gl.ylabel_style = {'fontsize':20}

        plt.legend(prop={'size': 18})
        plt.title('Subregion Filter', fontsize=30)
        plt.savefig(data_folder + 'subregion', dpi=300)


    # Making filenames to read predictions csvs
    days = [str(n).zfill(2) for n in range(1, 32)]
    times = ["09_20_00", "12_20_00"]

    # Since the predictions take up so much space, they are separated by day
    prediction_sets_aug = ["predictions_08_" + day + "_17_" + time for day in days for time in times] ###
    prediction_sets_aug

    days = [str(n).zfill(2) for n in range(1, 31)]
    times = ["09_20_00", "12_20_00"]

    # September files
    prediction_sets_sept = ["predictions_09_" + day + "_17_" + time for day in days for time in times] ###
    prediction_sets_sept

    prediction_sets = prediction_sets_aug + prediction_sets_sept
    prediction_sets = prediction_sets[:28] # this line is temporary

    # Emulator Evaluation
    #making empty data storage lists for last calculations
    which_gets_least_squares = []
    distances = []
    variances = []

    num_variants = inputs_df.shape[0]

    if toggle_filter:
        subregion_filt_idx_set = list((my_obs_df['latitude'] < lat_low) | (my_obs_df['longitude'] < lon_low) | (my_obs_df['latitude'] > lat_high) | (my_obs_df['longitude'] > lon_high))
        my_obs_df.loc[subregion_filt_idx_set , ['meanResponse', 'sdResponse']] = [float("nan"), float("nan")]

    my_obs_df.loc[my_obs_df.sdResponse >= obsSdCensor, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]



    for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        print(tm,prediction_set)
        my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index) # give the number of lat_long points for one time
        
        #need to make df of prediction outputs for each day
        my_predict_df_this_time = pd.read_csv(ocean_smokeppe_dir + 'predictions/' + prediction_set + '.csv', index_col=0) ###
        my_predict_df_this_time.sort_values(['longitude','latitude','variant'],inplace=True, ignore_index=True)
        
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
            zs = my_predict_dfs[row]['mean']
            ss = my_predict_dfs[row]['std']**2

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
        print(f'Done with {prediction_set}')

    leastSqs = [np.abs(distances[k]) / np.sqrt(variances[k]) for k in range(len(distances))]


    #Here is where the sorting becomes very important as the distances and variances append to the incorrect corresponding gstp
    outliers_df = my_obs_df
    outliers_df['leastSquares'] = leastSqs
    outliers_df['distances'] = distances
    outliers_df['variances'] = variances
    outliers_df.to_csv(save_here_dir + 'outliers.csv',index=True)


if __name__ == '__main__':
    runEmEval()
