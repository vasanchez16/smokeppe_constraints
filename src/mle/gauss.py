import pandas as pd
import numpy as np
import scipy
import scipy.stats
from scipy.optimize import minimize_scalar, minimize
import sys
import os
import json
from tqdm import tqdm
from src.storage.utils import save_dataset
sys.path.append(os.getcwd())


def mle_gauss(args, distances, variances, num_variants):
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']
    
    # Define function to be used in minimize scalar
    def minus_log_l(d):
        sigma_opt = d[0]
        if len(init_vals) > 1:
            epsilon = d[1]
        else:
            epsilon = 0
        # get dists for one param set
        dists = distances.iloc[:,param_set]
        varis = variances.iloc[:,param_set]
        # Log likelihood, to be maximized
        term1 = np.nansum(np.log(varis + sigma_opt**2)) #get all the gspts for emulation variant
        term2 = np.nansum(np.power(dists + epsilon, 2) / (varis + sigma_opt**2))
        return 0.5 * (term1 + term2)

    # Run minimize scalar for each parameter set
    max_l_for_us = []
    sigma_sqr_terms = []
    epsilon_terms = []
    progress_bar = tqdm(total=num_variants, desc="Progress")
    for u in range(num_variants):
        param_set = u
        x_0 = init_vals
        if len(init_vals) > 1:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])])
            epsilon_terms.append(res.x[1])
        else:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0])])
        max_l_for_us.append(-res.fun)
        sigma_sqr_terms.append(res.x[0]**2)
        progress_bar.update(1)
    progress_bar.close()

    if len(init_vals) > 1:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,epsilon_terms], index = ['log_L', 'sigma_sqr', 'epsilon']).transpose()
    else:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms], index = ['log_L', 'sigma_sqr']).transpose()
    save_dataset(all_mle, save_here_dir + 'all_mle.csv')

    # Find parameter set that gives the max likelihood
    u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # Use this parmeter set to get the model discrep term at that parameter set
    param_set = u_mle
    x_0 = init_vals
    if len(init_vals) > 1:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])]).x #val for model discrep term
    else:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0])]).x #val for model discrep term

    sigma = dec_vars[0]
    sigma_sqr = sigma**2
    column_names = ['parameter_set_num', 'variance_mle']
    optimized_vals = [u_mle, sigma_sqr]

    if len(init_vals) > 1:
        epsilon = dec_vars[1]
        column_names = column_names + ['epsilon']
        optimized_vals = optimized_vals + [epsilon]

    return optimized_vals, column_names
