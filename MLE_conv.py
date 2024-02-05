import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import os
from convolution import conv_gauss_t
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

Likelihoods = []
sigma_t = []
sigma_t_sqr = []
nu_t = []

# for col in range(my_distances.shape[1]):
for col in [0,1]:
    dists_here = my_distances.iloc[:,col].values
    varis_here = my_variances.iloc[:,col].values

    conv_here = conv_gauss_t(dists_here,varis_here)

    opt_res_here = conv_here.opt_this()

    Likelihoods.append(opt_res_here.fun * -1)
    sigma_t.append(opt_res_here.x[0])
    sigma_t_sqr.append(opt_res_here.x[0]**2)
    nu_t.append(opt_res_here.x[1])
    print(f'Done with {col}')

mle_stats = pd.DataFrame([Likelihoods,sigma_t,sigma_t_sqr,nu_t],index=['L','sigma_t','sigma_t_sqr','nu_t']).transpose()

mle_stats.to_csv(data_folder + 'mle_stats',index=True)