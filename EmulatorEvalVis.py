import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import imageio.v2 as imageio

with open('run_label.txt', 'r') as file:
    run_label = file.read()
print(f'Run label: {run_label}')
ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'

outs_df = pd.read_csv(data_folder + 'outliers.csv', index_col=0)
outs_df.sort_values(['time','latitude','longitude'], inplace=True, ignore_index=True)
outs_df.loc[outs_df['outlier'], ['leastSquares']] = float('nan')

ls_rs = np.reshape(
    np.array(outs_df['leastSquares']),
    ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
           )

missing_rs = np.reshape(
    np.array(outs_df['missing']),
    ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
)

outliers_rs = np.reshape(
    np.array(outs_df['outlier']),
    ( len(outs_df['time'].unique()) , len(outs_df['latitude'].unique()), len(outs_df['longitude'].unique()))
)

png_save_dir = 'outliersVisFigs/'
movieName = 'OutliersVis'

os.makedirs(data_folder+png_save_dir, exist_ok=True)

j=10
BBox = [
            np.min(outs_df.longitude) - 5, 
            np.max(outs_df.longitude) + 5,
            np.min(outs_df.latitude) - 5, 
            np.max(outs_df.latitude) + 5
               ]

missing_cmap = ListedColormap(['none', 'red'])
outlier_cmap = ListedColormap(['none', 'green'])

file_num = 0
for j in range(len(ls_rs)):
    fig = plt.figure(figsize=(20,12))
    projection = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=projection)
    ax.coastlines()
    ax.set_extent(BBox, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=projection, color='k', linewidth=0.25)
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

    plt.pcolormesh(
        outs_df['longitude'].unique(),
        outs_df['latitude'].unique(),
        outliers_rs[j],
        cmap=outlier_cmap,
        label='Outlier'
    )

    plt.pcolormesh(
        outs_df['longitude'].unique(),
        outs_df['latitude'].unique(),
        ls_rs[j],
        # vmin=0,
        # vmax=1,
        cmap='Oranges'
    )

    plt.colorbar(label='Minimum Least Squares')

    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Missing Data'),
        Patch(facecolor='green', edgecolor='green', label='Outlier')
                    ]

    # Create the custom legend
    ax.legend(handles=legend_elements,bbox_to_anchor=(0.65, -0.005), fontsize=20)

    plt.title(outs_df['time'].unique()[j])

    plt.savefig(data_folder+png_save_dir+f'outlierfig{file_num}.png',format='png')
    plt.cla()
    plt.clf()
    plt.close(fig)
    file_num+=1

# make mp4 file
frames = []
for i in range(file_num):
    frames.append(imageio.imread(data_folder+ png_save_dir + f'outlierfig{i}.png'))

imageio.mimsave(data_folder + movieName + '.mp4', frames, fps=5)