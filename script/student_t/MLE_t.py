import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
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
def minus_log_l(d):
    # get dists for one param set
    dists = my_distances.iloc[:,param_set]
    varis = my_variances.iloc[:,param_set]
    # Log likelihood, to be maximized
    sigma_opt = d[0]
    nu_opt = d[1]

    coeff = scipy.special.gamma((nu_opt + 1) / 2) / (scipy.special.gamma(nu_opt/2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))

    factor2 = 1 + (dists**2) / ((varis + sigma_opt**2) * (nu_opt-2))

    f_t = coeff * factor2**(-1*(nu_opt+1)/2)

    log_Li = np.log(f_t)
    log_likelihood = np.nansum(log_Li)
    return -1*log_likelihood

# Run minimize scalar for each parameter set
max_l_for_us = []
sigma_sqr_terms = []
nu_terms = []
for u in range(num_variants):
    param_set = u
    if u%1000 == 0:
        print(f'Parameter set: {u}')
    x_0 = [0.02,5]
    res = minimize(minus_log_l,x_0,bounds=[(0,1),(2+1E-5,30)])
    max_l_for_us.append(-res.fun)
    sigma_sqr_terms.append(res.x[0]**2)
    nu_terms.append(res.x[1])

# Find parameter set that gives the max likelihood
u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
# Use this parmeter set to get the model discrep term at that parameter set
param_set = u_mle
x_0 = [0.02,5]
dec_vars = minimize(minus_log_l,x_0,bounds=[(0,1),(2+1E-5,30)]).x #val for model discrep term
sigma = dec_vars[0]
sigma_sqr = sigma**2
nu = dec_vars[1]

###############################################################################
# Save metrics to dataframe and csv
mle_df = pd.DataFrame([u_mle,sigma_sqr,nu],index=['parameterSetNum','variance_mle','nu']).transpose()
mle_df.to_csv(data_folder + 'mle.csv',index=True)

# Save all likelihood terms and all model discrep terms to dataframe
param_mle_stats_df = pd.DataFrame([sigma_sqr_terms, nu_terms, max_l_for_us], index=['sigma_sqr', 'nu', 'likelihood']).transpose()
param_mle_stats_df.to_csv(data_folder + 'param_mle_stats.csv',index=True)
print('-------------------------------------------------------------------------------------')