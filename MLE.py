import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
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
inputs_df = pd.read_csv(ocean_smokeppe_dir + 'emulatorVariants10k.csv',index_col=0) ###
num_variants = inputs_df.shape[0]
###############################################################################

print('Reading in dists...')
my_distances = pd.read_csv(data_folder + 'distances.csv',index_col=0)
print('Reading in varis...')
my_variances = pd.read_csv(data_folder + 'variances.csv',index_col=0)

###############################################################################
# Define function to be used in minimize scalar
def l(d, u):
    # Log likelihood, to be maximized
    term1 = np.nansum(np.log(my_variances.iloc[:, u] + d**2)) #get all the gspts for emulation variant
    term2 = np.nansum(np.power(my_distances.iloc[:, u], 2) / (my_variances.iloc[:, u] + d**2))
    return 0.5 * (term1 + term2)

# Run minimize scalar for each parameter set
max_l_for_us = []
model_discrep_terms = []
for u in range(num_variants):
    if u%1000 == 0:
        print(f'Parameter set: {u}')
    res = minimize_scalar(l, args=(u))
    max_l_for_us.append(-res.fun)
    model_discrep_terms.append(res.x**2)

# Find parameter set that gives the max likelihood
u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
# Use this parmeter set to get the model discrep term at that parameter set
additional_variance = minimize_scalar(l, args=(u_mle)).x**2 #val for model discrep term
# additional_variance = np.median(model_discrep_terms) **2
# additional_variance = 0.025174600244882047 #defsimmle

###############################################################################
# Save metrics to dataframe and csv
mle_df = pd.DataFrame([u_mle,additional_variance],index=['parameterSetNum','modelDiscrep']).transpose()
mle_df.to_csv(data_folder + 'mle.csv',index=True)

# Save all likelihood terms and all model discrep terms to dataframe
param_mle_stats_df = pd.DataFrame([model_discrep_terms, max_l_for_us], index=['delta_mle', 'likelihood']).transpose()
param_mle_stats_df.to_csv(data_folder + 'param_mle_stats.csv',index=True)
print('-------------------------------------------------------------------------------------')