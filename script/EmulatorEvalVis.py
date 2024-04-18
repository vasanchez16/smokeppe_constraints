import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import imageio.v2 as imageio
import json
import configparser
import argparse

# Config
config = configparser.ConfigParser()
config.read('config.ini')

input_file = config.get('DEFAULT', 'InputFile')
output_dir = config.get('DEFAULT', 'OutputDir')

def EmulatorEvalVis(args):
    """
    Visualizes spatially where outliers tend to occur
    """
    print('---------EmulatorEval---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    save_figs_dir = save_here_dir + 'figures/'
    ls_thresh = eval_params['leastSquaresThreshold']

    outs_df = pd.read_csv(save_here_dir + 'outliers.csv')
    outs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)
    # outs_df.loc[outs_df['outlier'], ['leastSquares']] = float('nan')

    ls_rs = np.reshape(
        np.array(outs_df['leastSquares']),
        ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
            )

    missing_rs = np.reshape(
        np.array(outs_df['missing']),
        ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
    )

    if not os.path.exists(save_here_dir + 'movieFigs/'):
        os.mkdir(save_here_dir + 'movieFigs/')

    # outliers_rs = np.reshape(
    #     np.array(outs_df['outlier']),
    #     ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
    # )

    BBox = [
                np.min(outs_df.longitude) - 5, 
                np.max(outs_df.longitude) + 5,
                np.min(outs_df.latitude) - 5, 
                np.max(outs_df.latitude) + 5
                ]

    missing_cmap = ListedColormap(['none', 'black'])
    outlier_cmap = ListedColormap(['none', 'green'])
    vmax_here = ls_thresh + ls_thresh*.05

    file_num = 0
    for j in range(len(ls_rs)):
        fig = plt.figure(figsize=(20,12))
        projection = ccrs.PlateCarree()

        ax = fig.add_subplot(111, projection=projection)
        ax.coastlines()
        ax.set_extent(BBox, ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, crs=projection, color='k', linewidth=0.25, alpha=0.5)
        gl.bottom_labels=False
        gl.right_labels=False
        gl.ylocator=ticker.FixedLocator(list(range(-90,91,10)))
        gl.ylabel_style = {'size': 7}
        gl.xlocator=ticker.FixedLocator(list(range(-180,181,20)))
        gl.xlabel_style = {'size': 7}


        plt.pcolormesh(
            outs_df['longitude'].unique(),
            outs_df['latitude'].unique(),
            missing_rs[j],
            cmap=missing_cmap,
            label='Missing'
        )

        # plt.pcolormesh(
        #     outs_df['longitude'].unique(),
        #     outs_df['latitude'].unique(),
        #     outliers_rs[j],
        #     cmap=outlier_cmap,
        #     label='Outlier'
        # )

        plt.pcolormesh(
            outs_df['longitude'].unique(),
            outs_df['latitude'].unique(),
            ls_rs[j],
            # vmin=0,
            vmax=vmax_here,
            cmap='Reds'
        )

        plt.colorbar(label='Minimum Least Squares')

        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Missing Data')
                        ]
        
        # legend_elements = [
        #     Patch(facecolor='red', edgecolor='red', label='Missing Data'),
        #     Patch(facecolor='green', edgecolor='green', label='Outlier')
        #                 ]

        # Create the custom legend
        ax.legend(handles=legend_elements,bbox_to_anchor=(0.65, -0.005), fontsize=20)

        plt.title(outs_df['time'].unique()[j])

        plt.savefig(save_here_dir + 'movieFigs/' + f'outlierfig{file_num}.png',format='png')
        plt.cla()
        plt.clf()
        plt.close(fig)
        file_num+=1

    # make mp4 file
    frames = []
    for i in range(file_num):
        frames.append(imageio.imread(save_here_dir + 'movieFigs/' + f'outlierfig{i}.png'))

    imageio.mimsave(save_here_dir + 'outliersVis' + '.mp4', frames, fps=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument("--savefigs", action="store_true", default=False)
    parser.add_argument("--laplace", action="store_true", default=False)
    parser.add_argument(
        "--input_file",
        type=str,
        default=input_file
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir
    )
    args = parser.parse_args()
    EmulatorEvalVis(args)