import numpy as np
import pandas as pd
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

mult_targets = False
###############################################################################
# Read in necessary statistics
mle_df = pd.read_csv(data_folder + 'mle.csv',index_col=0)
mle_param_set_num = int(mle_df.iloc[0,0])
additional_variance = mle_df['variance_mle'].values[0]

print('Reading in dists...')
my_distances = pd.read_csv(data_folder + 'distances.csv',index_col=0)
print('Reading in varis...')
my_variances = pd.read_csv(data_folder + 'variances.csv',index_col=0)
my_variances_adjusted = my_variances + additional_variance

# if mult_targets:
    # import multiple MLE's
    # mle_aod = 
    # mle_clwp = 

    # Read in distances and variances
    # my_distances_aod = 
    # my_distances_clwp =
    
    # my_variances_aod = 
    # my_variances_clwp =

    # Adjust variances
    # my_variances_adjusted_aod = my_variances_aod + mle_aod
    # my_variances_adjusted_clwp = my_variances_adjusted_clwp + mle_clwp

    # Calculate Impluasibilites
    # aod_implaus_terms = np.power(my_distances_aod, 2).div(my_variances_adjusted_aod).sum(axis=0)
    # clwp_implaus_terms = np.power(my_distances_clwp, 2).div(my_variances_adjusted_clwp).sum(axis=0)
    # implausibilities = np.sqrt(
    #     aod_implaus_terms + clwp_implaus_terms
    #                            )
    # None
# Calculate Impluasibility quantities for every parameter set
implausibilities = np.sqrt(np.power(my_distances, 2).div(my_variances_adjusted).sum(axis=0))
# Save Implausibility values
implausibilities.to_csv(data_folder + 'implausibilities.csv', index=True)
print('-------------------------------------------------------------------------------------')