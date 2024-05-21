import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import ticker
from matplotlib.colors import ListedColormap
import os
import imageio.v2 as imageio
import json
from tqdm import tqdm

def plot_measurements(lon_min,
                      lat_min,
                      lon_max,
                      lat_max,
                      snip_subregion_obs,
                      save_here_dir):
    """
    """
    plt.figure(figsize=(12,10))
    ax = plt.subplot(111,projection=ccrs.PlateCarree())
    plt.gca().coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', color='black', alpha=0.5)

    rect1 = patches.Rectangle((lon_min, lat_min), 
                            lon_max - lon_min, 
                            lat_max - lat_min, 
                            linestyle='--', edgecolor='red', facecolor='none', linewidth=4, label='Subregion'
                            )
    ax.add_patch(rect1)

    plt.scatter(
        snip_subregion_obs['longitude'],
        snip_subregion_obs['latitude'],
        color='black'
    )

    ax.add_feature(cfeature.LAND, facecolor='tan')
    ax.add_feature(cfeature.LAKES, alpha=0.9, zorder=-1)  
    ax.add_feature(cfeature.BORDERS, zorder=-1)
    ax.add_feature(cfeature.COASTLINE, zorder=-1)
    ax.add_feature(cfeature.OCEAN,alpha=0.5)

    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}

    plt.legend(prop={'size': 18}, loc=1)
    plt.title('Subregion Filter', fontsize=30)
    plt.savefig(save_here_dir + 'general_figures/' + 'subregion.png', dpi=300)
    return


def plot_outliers(outliers_df, ls_thresh, save_here_dir):
    # Initiate for plots
    fig, ax1 = plt.subplots(figsize=(10,10))
    # Plot outer histogram
    bin_info = ax1.hist(outliers_df.leastSquares,bins=400)
    bin_sizes = bin_info[0]
    max_bin_size = max(bin_sizes)
    # Set inner axes
    axins = ax1.inset_axes([0.25,0.28,0.7,.7])
    # Plot inner axes
    axins.hist(outliers_df.leastSquares,bins=400)
    # Create threshold line on plot
    axins.vlines(ls_thresh,0,max_bin_size*2,linestyles='--',color='red')
    ax1.vlines(ls_thresh,0,max_bin_size*2,linestyles='--',color='red')

    #subregion of original image
    x1 = -1
    x2 = np.nanpercentile(outliers_df.leastSquares,99.95)
    y1 = 0
    y2 = max_bin_size * 0.1
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    x1 = -1
    x2 = np.nanmax(outliers_df.leastSquares)
    y1 = 0
    y2 = max_bin_size + (max_bin_size * 0.05)
    ax1.set_xlim(x1,x2)
    ax1.set_ylim(y1,y2)    
    # Initiate
    ax1.indicate_inset_zoom(axins, edgecolor='black')
    # Labels
    plt.xlabel('Minimum Least Squares Value for Each Surrogate Model', fontsize=14)
    plt.ylabel('Number of Occurences', fontsize=14)
    plt.savefig(save_here_dir + 'general_figures/' + 'outliersFig', dpi = 300)
    return

def emulator_eval_vis(args):
    """
    Visualizes spatially where outliers tend to occur
    """
    print('---------EmulatorEvalVis---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    outs_df = pd.read_csv(save_here_dir + 'outliers.csv')
    outs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)

    file_num = 0
    progress_bar = tqdm(total=len(np.unique(outs_df['time'])), desc="Progress")
    for time_now in np.unique(outs_df['time']):
        # get df for only one time
        outs_now = outs_df[ outs_df['time'] == time_now ]
        # save rows that are outlier points
        outlier_only_now = outs_now[ outs_now['outlier'] ]

        fig = plt.figure(figsize=(20,12))
        projection = ccrs.PlateCarree()

        ax = fig.add_subplot(111, projection=projection)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, crs=projection, color='k', linewidth=0.25, alpha=0.5)
        gl.ylocator=ticker.FixedLocator(list(range(-90,91,10)))
        gl.xlocator=ticker.FixedLocator(list(range(-180,181,20)))
        gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}

        ax.add_feature(cfeature.LAND, facecolor='tan')
        ax.add_feature(cfeature.LAKES, alpha=0.9, zorder=-1)  
        ax.add_feature(cfeature.BORDERS, zorder=-1)
        ax.add_feature(cfeature.COASTLINE, zorder=-1)
        ax.add_feature(cfeature.OCEAN,alpha=0.5)

        # plot the least squares value of each point
        plt.scatter(
            outs_now['longitude'],
            outs_now['latitude'],
            c = outs_now['leastSquares'],
            cmap = 'hot_r',
            edgecolor = 'black',
            vmax = np.nanpercentile(outs_df['leastSquares'],99)
        )
        cbar = plt.colorbar()

        # mark the missing data
        missing_color = 'blue'
        missing_cmap = ListedColormap(['none', missing_color])
        plt.scatter(
            outs_now['longitude'],
            outs_now['latitude'],
            c = outs_now['missing'],
            cmap = missing_cmap,
            edgecolor = 'black'
        )

        # mark the outlier data
        plt.scatter(
            outlier_only_now['longitude'],
            outlier_only_now['latitude'],
            facecolor = 'None',
            edgecolor = 'red',
        )

        legend_elements = [
            patches.Patch(facecolor=missing_color, edgecolor='black', label='Missing Points'),
            patches.Patch(facecolor='None', edgecolor='red', label='Outlier Points')
                        ]

        # Create the custom legend
        ax.legend(handles=legend_elements,bbox_to_anchor=(0.65, -0.05), fontsize=20)

        plt.title(time_now)

        plt.savefig(save_here_dir + 'general_figures/movie_pngs/' + f'outlierfig{file_num}.png',format='png')
        plt.cla()
        plt.clf()
        plt.close(fig)
        file_num+=1
        progress_bar.update(1)
    progress_bar.close()

    # make mp4 file
    frames = []
    for i in range(file_num):
        frames.append(imageio.imread(save_here_dir + 'general_figures/movie_pngs/' + f'outlierfig{i}.png'))
    print('Saving mp4 file...')
    imageio.mimsave(save_here_dir + 'general_figures/outliersVis' + '.mp4', frames, fps=5)