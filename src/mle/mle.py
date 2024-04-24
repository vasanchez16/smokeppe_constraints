import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.storage.utils import save_dataset, save_indexed_dataset
from .gauss import mle_gauss
from .student_t import mle_t
import json


def mle(args, my_distances, my_variances):
    """
    Collect datasets
    """
    print('---------MLE---------')
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']

    inputs_file_path = eval_params['emulator_inputs_file_path']

    inputs_df = pd.read_csv(inputs_file_path)
    num_variants = inputs_df.shape[0]

    # print('Reading in distances...')
    # my_distances = pd.read_csv(save_here_dir + 'distances.csv',index_col=0)
    # print('Reading in variances...')
    # my_variances = pd.read_csv(save_here_dir + 'variances.csv',index_col=0)


    """
    Calculate MLE for model discrepancy
    """
    if stats_dist_method == 'convolution':
       opt_vals,col_names = approx_mle(my_distances, my_variances, num_variants, args.laplace)
    elif stats_dist_method == 'student-t':
        opt_vals,col_names = mle_t(args, my_distances, my_variances, num_variants)
    elif stats_dist_method == 'gaussian':
        opt_vals,col_names = mle_gauss(args, my_distances, my_variances, num_variants)
        # add to this for gaussian if necessary
        # x_0,x_1,fun_val = approx_mle(my_distances, my_variances, num_variants, args.laplace)

    """
    Save datasets
    """
    # Save metrics to dataframe and csv
    mle_df = pd.DataFrame(opt_vals,index=col_names).transpose()
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