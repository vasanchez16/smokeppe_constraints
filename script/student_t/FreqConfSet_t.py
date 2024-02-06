import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
from matplotlib.colors import ListedColormap
import os
current_script_path = os.path.abspath(__file__)
current_script_name = os.path.basename(current_script_path)
print(current_script_name)

with open('run_label.txt', 'r') as file:
    run_label = file.read()
print(f'Run label: {run_label}')
ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'

implausibilites = pd.read_csv(data_folder + 'implausibilities.csv', index_col=0)
inputs_df = pd.read_csv(ocean_smokeppe_dir + 'emulatorVariants10k.csv', index_col=0) ###

obs_df = pd.read_csv(data_folder + 'outliers.csv', index_col = 0)

mle_df = pd.read_csv(data_folder + 'mle.csv',index_col=0)
nu_opt = mle_df['nu'].values[0]

param_short_names = list(inputs_df.columns)

num_points = sum(~obs_df['missing'] & ~obs_df['outlier'])
sum_this = []
for i in range(num_points):
    test_t = np.random.standard_t(nu_opt,10000)
    sum_this.append(test_t)
sum_this = [i * i for i in sum_this]
summed = np.sum(sum_this,axis=0)
sb_cv = np.sqrt(np.percentile(summed,95))

print(f'Threshold for 95th percentile: {sb_cv}')

param_dict = {
    'smoke_emiss' : 'Smoke Emissions',
    'smoke_diam': 'Smoke Diameter',
    'sig_w': 'Standard Deviation of the Updraft Velocity',
    'sea_spray': 'Sea Spray Emissions',
    'kappa_oc': 'Kappa-Kohler Coeff. for Organic Carbon',
    'dry_dep_acc': 'Dry Deposition of Accumulation Mode Aerosol',
    'dms': 'Dimethyl Sulfide Ocean Surface Concentration',
    'bparam': 'Beta Parameter',
    'bc_ri': 'Black Carbon Refractive Index',
    'autoconv_exp_nd': 'Autoconversion Exponent',
    'anth_so2': 'Anthropogenic SO_2',
    'a_ent_1_rp': 'Cloud Top Entrainment Rate',
    'acure_bl_nuc': 'acure bl nuc',
    'acure_ait_width': 'acure ait width',
    'acure_cloud_ph': 'acure cloud ph',
    'acure_carb_bb_diam': 'acure carb bb diam',
    'acure_prim_so4_diam': 'acure prim so4 diam',
    'acure_sea_spray': 'acure sea spray',
    'acure_anth_so2_r': 'acure anth so2 r',
    'acure_bvoc_soa': 'acure bvoc soa',
    'acure_dms': 'acure dms',
    'acure_dry_dep_ait': 'acure dry dep ait',
    'acure_dry_dep_acc': 'acure dry dep acc',
    'acure_dry_dep_so2': 'acure dry dep so2',
    'acure_bc_ri': 'acure bc ri',
    'acure_autoconv_exp_nd': 'acure autoconv exp nd',
    'dbsdtbs_turb_0': 'dbsdtbs turb 0'
}


my_input_df = inputs_df.copy()


my_input_df['implausibilities'] = implausibilites
title = 'Strict bounds implausibilities'
cv = sb_cv


my_input_df['colors'] = (my_input_df['implausibilities']>cv)


if not os.path.exists(data_folder + 'implausFigs' + run_label):
    os.mkdir(data_folder + 'implausFigs' + run_label)


custom_cmap = ListedColormap(['blue', 'red'])


for param in param_short_names:
    fig = plt.figure(facecolor='white',dpi=1200)
    

    plt.scatter(
        my_input_df[param],
        my_input_df['implausibilities'],
        alpha=1,
        s=0.01,
        c=my_input_df['colors'],
        cmap=custom_cmap
    )
    plt.axhline(
        cv,
        c='r',
        label = 'Implausibility Threshold'
    )

    plt.xlabel(param_dict[param], fontsize=8)
    plt.ylabel(r'$I(u^k)$', fontsize = 20)
    plt.ylim([0,max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities'])])
    plt.legend()

    plt.savefig(data_folder + 'implausFigs' + run_label + '/' + param, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)
