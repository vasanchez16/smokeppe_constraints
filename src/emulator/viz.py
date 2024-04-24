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
    plt.savefig(save_here_dir + 'subregion.png', dpi=300)
    return


def plot_outliers(outliers_df, ls_thresh, save_here_dir):
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
    return
