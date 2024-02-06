import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
current_script_path = os.path.abspath(__file__)
current_script_name = os.path.basename(current_script_path)
print(current_script_name)

with open('run_label.txt', 'r') as file:
    run_label = file.read()
print(f'Run label: {run_label}')

# Create variables to save file paths
ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'

###############################################################################
ls_thresh = 8
###############################################################################

outliers_df = pd.read_csv(data_folder + 'outliers.csv',index_col=0)

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
plt.savefig(data_folder + 'outliersFig', dpi = 300)
# Create column to label missing data
outliers_df['missing'] = np.isnan(outliers_df.leastSquares)


print(f'Least squares threshold:{ls_thresh}')
# Create column to label points as outliers or above LS threshold
outliers_df['outlier'] = [
    outliers_df.leastSquares[k] > ls_thresh for k in range(len(outliers_df.leastSquares))
                          ]
# Save new outliers csv with outlier and missing columns
outliers_df.to_csv(data_folder + 'outliers.csv', index=True)
print('-------------------------------------------------------------------------------------')