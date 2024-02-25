import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as patches 
import os
from datetime import datetime
import json
from src.inference.utils import get_em_pred_filenames


def EmulatorEval(args):
    """
    input_file
    output_dir
    """
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label

    emulator_folder_path = eval_params['emulator_output_folder_path']
    satellite_file_path = eval_params['satellite_file_path']
    obsSdCensor = eval_params['obsSdCensor']
    ls_thresh = eval_params['leastSquaresThreshold']
    inputs_file_path = eval_params['emulator_inputs_file_path']
    subregion_filter = eval_params['subregion_filter']

    # Extracting values from the subregion_filter dictionary
    toggle_filter = subregion_filter['toggle_filter']
    lat_min = subregion_filter['lat_min']
    lat_max = subregion_filter['lat_max']
    lon_min = subregion_filter['lon_min']
    lon_max = subregion_filter['lon_max']

    # Import input emulator parameter combinations
    inputs_df = pd.read_csv(inputs_file_path,index_col=0)

    # Import MODIS observations dataframe
    my_obs_df = pd.read_csv(satellite_file_path,index_col=0)
    my_obs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)

    prediction_sets = get_em_pred_filenames(args)
    prediction_sets = prediction_sets[:28] # this line is temporary

    # Emulator Evaluation
    # making empty data storage lists for last calculations
    which_gets_least_squares = []
    distances = []
    variances = []

    num_variants = inputs_df.shape[0]

    my_obs_df.loc[my_obs_df.sdResponse >= obsSdCensor, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]

    for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        print(tm,prediction_set)
        my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index) # give the number of lat_long points for one time
        
        #need to make df of prediction outputs for each day
        my_predict_df_this_time = pd.read_csv(emulator_folder_path + prediction_set + '.csv', index_col=0) ###
        my_predict_df_this_time.sort_values(['latitude','longitude','variant'],inplace=True, ignore_index=True)
        
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
        print(f'{prediction_set} completed.')

    leastSqs = [np.abs(distances[k]) / np.sqrt(variances[k]) for k in range(len(distances))]


    #Here is where the sorting becomes very important as the distances and variances append to the incorrect corresponding gstp
    outliers_df = my_obs_df
    outliers_df['leastSquares'] = leastSqs
    outliers_df['distances'] = distances
    outliers_df['variances'] = variances

    #filter out the outliers

    # Initiate for plots
    fig, ax1 = plt.subplots(figsize=(10,10))
    # Plot outer histogram
    ax1.hist(outliers_df.leastSquares,bins=400)
    # Set inner axes
    axins = ax1.inset_axes([0.25,0.28,0.7,.7])
    # Plot inner axes
    axins.hist(outliers_df.leastSquares,bins=400)
    # Create threshold line on plot
    axins.vlines([ls_thresh,ls_thresh],-1,2501,linestyles='--',color='red')

    #subregion of original image
    x1,x2,y1,y2 = -5,40,0,2500
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    # Initiate
    ax1.indicate_inset_zoom(axins, edgecolor='black')
    # Labels
    plt.xlabel('Minimum Least Squares Value for Each Surrogate Model', fontsize=14)
    plt.ylabel('Number of Occurences', fontsize=14)
    plt.savefig(save_here_dir + 'outliersFig', dpi = 300)
    # Create column to label missing data


    outliers_df['missing'] = np.isnan(outliers_df.leastSquares)
    print(f'Least squares threshold:{ls_thresh}')
    # Create column to label points as outliers or above LS threshold
    outliers_df['outlier'] = [
        outliers_df.leastSquares[k] > ls_thresh for k in range(len(outliers_df.leastSquares))
                            ]
    # Save new outliers csv with outlier and missing columns
    outliers_df.to_csv(save_here_dir + 'outliers.csv', index=True)


# Formerly EmulatorEvalVis

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from matplotlib import ticker
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch
# import os
# import imageio.v2 as imageio

# with open('run_label.txt', 'r') as file:
#     run_label = file.read()
# print(f'Run label: {run_label}')
# ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
# data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'

# outs_df = pd.read_csv(data_folder + 'outliers.csv', index_col=0)
# outs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)
# outs_df.loc[outs_df['outlier'], ['leastSquares']] = float('nan')

# ls_rs = np.reshape(
#     np.array(outs_df['leastSquares']),
#     ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
#            )

# missing_rs = np.reshape(
#     np.array(outs_df['missing']),
#     ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
# )

# outliers_rs = np.reshape(
#     np.array(outs_df['outlier']),
#     ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
# )

# png_save_dir = 'outliersVisFigs/'
# movieName = 'OutliersVis'

# os.makedirs(data_folder+png_save_dir, exist_ok=True)

# j=10
# BBox = [
#             np.min(outs_df.longitude) - 5, 
#             np.max(outs_df.longitude) + 5,
#             np.min(outs_df.latitude) - 5, 
#             np.max(outs_df.latitude) + 5
#                ]

# missing_cmap = ListedColormap(['none', 'red'])
# outlier_cmap = ListedColormap(['none', 'green'])

# file_num = 0
# for j in range(len(ls_rs)):
#     fig = plt.figure(figsize=(20,12))
#     projection = ccrs.PlateCarree()

#     ax = fig.add_subplot(111, projection=projection)
#     ax.coastlines()
#     ax.set_extent(BBox, ccrs.PlateCarree())
#     gl = ax.gridlines(draw_labels=True, crs=projection, color='k', linewidth=0.25)
#     gl.bottom_labels=False
#     gl.right_labels=False
#     gl.ylocator=ticker.FixedLocator(list(range(-90,91,10)))
#     gl.ylabel_style = {'size': 7}
#     gl.xlocator=ticker.FixedLocator(list(range(-180,181,20)))
#     gl.xlabel_style = {'size': 7}


#     plt.pcolormesh(
#         outs_df['longitude'].unique(),
#         outs_df['latitude'].unique(),
#         missing_rs[j],
#         cmap=missing_cmap,
#         label='Missing'
#     )

#     plt.pcolormesh(
#         outs_df['longitude'].unique(),
#         outs_df['latitude'].unique(),
#         outliers_rs[j],
#         cmap=outlier_cmap,
#         label='Outlier'
#     )

#     plt.pcolormesh(
#         outs_df['longitude'].unique(),
#         outs_df['latitude'].unique(),
#         ls_rs[j],
#         # vmin=0,
#         # vmax=1,
#         cmap='Oranges'
#     )

#     plt.colorbar(label='Minimum Least Squares')

#     legend_elements = [
#         Patch(facecolor='red', edgecolor='red', label='Missing Data'),
#         Patch(facecolor='green', edgecolor='green', label='Outlier')
#                     ]

#     # Create the custom legend
#     ax.legend(handles=legend_elements,bbox_to_anchor=(0.65, -0.005), fontsize=20)

#     plt.title(outs_df['time'].unique()[j])

#     plt.savefig(data_folder+png_save_dir+f'outlierfig{file_num}.png',format='png')
#     plt.cla()
#     plt.clf()
#     plt.close(fig)
#     file_num+=1

# # make mp4 file
# frames = []
# for i in range(file_num):
#     frames.append(imageio.imread(data_folder+ png_save_dir + f'outlierfig{i}.png'))

# imageio.mimsave(data_folder + movieName + '.mp4', frames, fps=5)
