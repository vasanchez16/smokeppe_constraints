import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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