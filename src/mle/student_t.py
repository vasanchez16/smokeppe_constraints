import pandas as pd
import numpy as np
import netCDF4 as nc
import scipy
from scipy.special import gamma
import scipy.stats
from scipy.optimize import minimize_scalar, minimize
import sys
import os
import json
from tqdm import tqdm
from src.storage.utils import save_dataset
sys.path.append(os.getcwd())


def mle_t(args, num_variants):
    """
    MLE analysis using the student-t distribution approximation.

    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    def minus_log_l(d):
        
        # Log likelihood, to be maximized
        sigma_opt = d[0]
        nu_opt = d[1]
        if len(init_vals) > 2:
            epsilon = d[2]
        else:
            epsilon = 0

        coeff = scipy.special.gamma((nu_opt + 1) / 2) / (scipy.special.gamma(nu_opt/2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))

        factor2 = 1 + ((dists + epsilon)**2) / ((varis + sigma_opt**2) * (nu_opt-2))

        f_t = coeff * factor2**(-1*(nu_opt+1)/2)

        log_Li = np.log(f_t)
        log_likelihood = np.nansum(log_Li)
        return -1*log_likelihood
    
    # Run minimize scalar for each parameter set
    max_l_for_us = []
    sigma_sqr_terms = []
    nu_terms = []
    epsilon_terms = []
    progress_bar = tqdm(total=num_variants, desc="Progress")
    for u in range(num_variants):
        nc_file = nc.Dataset(save_here_dir + 'distances_variances.nc','r')
        dists = nc_file['distances'][:,:,:,u].flatten()
        varis = nc_file['variances'][:,:,:,u].flatten()

        x_0 = init_vals
        if len(init_vals) > 2:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1]),tuple(bnds[2])], method = 'Nelder-Mead')
            epsilon_terms.append(res.x[2])
        else:    
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])], method = 'Nelder-Mead')
        max_l_for_us.append(-res.fun)
        sigma_sqr_terms.append(res.x[0]**2)
        nu_terms.append(res.x[1])
        progress_bar.update(1)
    progress_bar.close()


    if len(init_vals) > 2:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,nu_terms,epsilon_terms], index = ['log_L', 'sigma_sqr', 'nu', 'epsilon']).transpose()
    else:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,nu_terms], index = ['log_L', 'sigma_sqr', 'nu']).transpose()
    save_dataset(all_mle, save_here_dir + 'all_mle.csv')

    # Find parameter set that gives the max likelihood
    u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # Use this parmeter set to get the model discrep term at that parameter set
    dists = nc_file['distances'][:,:,:,u_mle].flatten()
    varis = nc_file['variances'][:,:,:,u_mle].flatten()
    x_0 = init_vals
    if len(init_vals) > 2:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1]),tuple(bnds[2])], method = 'Nelder-Mead').x #val for model discrep term
    else:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])], method = 'Nelder-Mead').x #val for model discrep term
    sigma = dec_vars[0]
    sigma_sqr = sigma**2
    nu = dec_vars[1]
    column_names = ['parameter_set_num', 'variance_mle', 'nu']
    optimized_vals = [u_mle, sigma_sqr, nu]
    nc_file.close()
    if len(init_vals) > 2:
        epsilon = dec_vars[2]
        column_names = column_names + ['epsilon']
        optimized_vals = optimized_vals + [epsilon]

    return optimized_vals, column_names
