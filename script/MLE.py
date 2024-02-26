import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.inference.utils import save_dataset, save_indexed_dataset
from src.inference import approx_mle
import json

def MLE(args):
    """
    Collect datasets
    """
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']

    inputs_file_path = eval_params['emulator_inputs_file_path']

    inputs_df = pd.read_csv(inputs_file_path,index_col=0)
    num_variants = inputs_df.shape[0]

    print('Reading in distances...')
    my_distances = pd.read_csv(save_here_dir + 'distances.csv',index_col=0)
    print('Reading in variances...')
    my_variances = pd.read_csv(save_here_dir + 'variances.csv',index_col=0)


    """
    Calculate MLE for model discrepancy
    """
    x_0,x_1,fun_val = approx_mle(my_distances, my_variances, num_variants, args.laplace)

    # # Define function to be used in minimize scalar
    # def l(d, u):
    #     # Log likelihood, to be maximized
    #     term1 = np.nansum(np.log(my_variances.iloc[:, u] + d**2)) #get all the gspts for emulation variant
    #     term2 = np.nansum(np.power(my_distances.iloc[:, u], 2) / (my_variances.iloc[:, u] + d**2))
    #     return 0.5 * (term1 + term2)

    # # Run minimize scalar for each parameter set
    # max_l_for_us = []
    # model_discrep_terms = []
    # for u in range(num_variants):
    #     if u%1000 == 0:
    #         print(f'Parameter set: {u}')
    #     res = minimize_scalar(l, args=(u))
    #     max_l_for_us.append(-res.fun)
    #     model_discrep_terms.append(res.x**2)

    # # Find parameter set that gives the max likelihood
    # u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # # Use this parmeter set to get the model discrep term at that parameter set
    # additional_variance = minimize_scalar(l, args=(u_mle)).x**2 #val for model discrep term


    """
    Save datasets
    """
    # Save metrics to dataframe and csv
    mle_df = pd.DataFrame([x_0,x_1,fun_val],index=['x_0','x_1','fun_val']).transpose()
    save_dataset(mle_df, save_here_dir + 'mle.csv')

    # Save all likelihood terms and all model discrep terms to dataframe
    # param_mle_stats_df = pd.DataFrame([model_discrep_terms, max_l_for_us], index=['delta_mle', 'likelihood']).transpose()
    # save_dataset(param_mle_stats_df, save_here_dir + 'param_mle_stats.csv')

    return


# Formerly MLE_conv

# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize_scalar
# import os
# from convolution import conv_gauss_t
# current_script_path = os.path.abspath(__file__)
# current_script_name = os.path.basename(current_script_path)
# print(current_script_name)

# with open('run_label.txt', 'r') as file:
#     run_label = file.read()
# print(f'Run label: {run_label}')
# ocean_smokeppe_dir = '/ocean/projects/atm200005p/vsanchez/SmokePPEOutputs/'
# data_folder = ocean_smokeppe_dir + 'results_runs/' + run_label + '/'
# ###############################################################################
# inputs_df = pd.read_csv(ocean_smokeppe_dir + 'emulatorVariants10k.csv',index_col=0) ###
# num_variants = inputs_df.shape[0]
# ###############################################################################

# print('Reading in dists...')
# my_distances = pd.read_csv(data_folder + 'distances.csv',index_col=0)
# print('Reading in varis...')
# my_variances = pd.read_csv(data_folder + 'variances.csv',index_col=0)

# Likelihoods = []
# sigma_t = []
# sigma_t_sqr = []
# nu_t = []

# # for col in range(my_distances.shape[1]):
# for col in [0,1]:
#     dists_here = my_distances.iloc[:,col].values
#     varis_here = my_variances.iloc[:,col].values

#     conv_here = conv_gauss_t(dists_here,varis_here)

#     opt_res_here = conv_here.opt_this()

#     Likelihoods.append(opt_res_here.fun * -1)
#     sigma_t.append(opt_res_here.x[0])
#     sigma_t_sqr.append(opt_res_here.x[0]**2)
#     nu_t.append(opt_res_here.x[1])
#     print(f'Done with {col}')

# mle_stats = pd.DataFrame([Likelihoods,sigma_t,sigma_t_sqr,nu_t],index=['L','sigma_t','sigma_t_sqr','nu_t']).transpose()

# mle_stats.to_csv(data_folder + 'mle_stats',index=True)