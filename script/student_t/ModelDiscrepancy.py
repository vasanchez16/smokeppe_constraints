import numpy as np
import pandas as pd
import os
current_script_path = os.path.abspath(__file__)
current_script_name = os.path.basename(current_script_path)
print(current_script_name)

with open('run_label.txt', 'r') as file:
    run_label = file.read()
print(f'Run label: {run_label}')
ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'

###############################################################################
save_best = True
###############################################################################

# Import input emulator parameter combinations
inputs_df = pd.read_csv(ocean_smokeppe_dir + 'emulatorVariants10k.csv',index_col=0) ###

# Import MODIS observations dataframe
obs_df = pd.read_csv(data_folder + 'outliers.csv',index_col=0)


# Making filenames to read predictions csvs
days = [str(n).zfill(2) for n in range(1, 32)]
times = ["09_20_00", "12_20_00"]

# Since the predictions take up so much space, they are separated by day
prediction_sets_aug = ["predictions_08_" + day + "_17_" + time for day in days for time in times] ###


days = [str(n).zfill(2) for n in range(1, 31)]
times = ["09_20_00", "12_20_00"]

# Since the predictions take up so much space, they are separated by day
prediction_sets_sept = ["predictions_09_" + day + "_17_" + time for day in days for time in times] ###

prediction_sets = prediction_sets_aug + prediction_sets_sept
prediction_sets = prediction_sets[:28] # this line is temporary

idxSet=list((obs_df['missing']) | (obs_df['outlier'])) ###

"""
y = observed AOD
zs = emulated AODs (vector)

e = estimated instrument error standard deviation
ss = standard deviation of AOD emulation


Arguments

idxSet : Set of row indices which are to be excluded from analysis
method : Which method to use for estimating uncertainties, either 'sb' (strict bounds, our method) or 'hm' (history
    matching, based on Johnson et al. (2020))


Value

Tuple : "Distances" (differences in response) and "variances" (terms needed to normalize the distances)
"""
allDistances = []
allVariances = []

num_variants = inputs_df.shape[0]

my_obs_df = obs_df.copy()
#set missing or outlier values to nan
my_obs_df.loc[idxSet, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]

for tm, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
    #time is a datetime string in this case, but df here has time in hours as float
    my_obs_df_this_time = my_obs_df[my_obs_df.time==tm].reset_index(drop=True)
    num_pixels = len(my_obs_df_this_time.index)
    
    my_predict_df_this_time = pd.read_csv(ocean_smokeppe_dir + 'predictions/' + prediction_set + '.csv', index_col=0) ###
    my_predict_df_this_time.sort_values(['longitude','latitude','variant'],inplace=True, ignore_index=True)
    #opens csv data that stores emulated data for each point, csv's are labeled by time
    print(f'Read in {prediction_set}')
    
    my_predict_dfs = [
        my_predict_df_this_time.iloc[k*num_variants:(k+1)*num_variants, :].reset_index(drop=True)
        for k in range(num_pixels)
    ]
    
    for row in range(num_pixels):
        y = my_obs_df_this_time.loc[row, 'meanResponse']
        e = my_obs_df_this_time.loc[row, 'sdResponse']**2

        zs = my_predict_dfs[row]['mean']
        ss = my_predict_dfs[row]['std']**2

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
all_dists_df.to_csv(data_folder + 'distances.csv', index=True)
print('Saving varis...')
all_vars_df.to_csv(data_folder + 'variances.csv', index=True)

if save_best:
    print('Saving best parameter set dists and varis...')
    dists_8210 = all_dists_df.iloc[:,8210]
    vars_8210 = all_vars_df.iloc[:,8210]

    dists_8210.to_csv(data_folder + 'dists_8210.csv', index=True)
    vars_8210.to_csv(data_folder + 'vars_8210.csv', index=True)

print('-------------------------------------------------------------------------------------')