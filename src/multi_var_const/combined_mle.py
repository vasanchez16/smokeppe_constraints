import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import json
from tqdm import tqdm
from src.storage.utils import save_dataset


def comb_mle_t(args, run_dirs, num_variants):
    """
    mle analysis using the student t approximation
    """

    # get eval parameters
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # array containing distances and variances df's for all comp vars
    dist_comp_var_arr = []
    vari_comp_var_arr = []
    
    # array containing opt info for all comp vars
    init_vals_comp_var_arr = []
    bounds_comp_var_arr = []

    for dir in run_dirs:
        if dir[-1] != '/':
            dir = dir + '/'

        # append to list
        short_dir = dir.split('/')[-2]
        print(f'Reading in data for {short_dir}')
        dist_comp_var_arr.append(pd.read_csv(dir + 'distances.csv'))
        vari_comp_var_arr.append(pd.read_csv(dir + 'variances.csv'))

        #read in opt info
        with open(dir + 'evaluationParameter.json','r') as file:
            dir_eval_params = json.load(file)
        
        # append info to list
        bounds_comp_var_arr.append(dir_eval_params['MLE_optimization']['bounds'])
        init_vals_comp_var_arr.append(dir_eval_params['MLE_optimization']['initial_vals'])

    def minus_log_l(d):
        """
        Objective function
        """
        # get data for one param set
        dists = []
        varis = []
        for i in range(len(run_dirs)):
            dists.append(dist_comp_var_arr[i].iloc[:,param_set])
            varis.append(vari_comp_var_arr[i].iloc[:,param_set])

        log_likelihood = 0
        for i in range(len(run_dirs)):
            
            # label optimization variables
            opt_var_num = 0
            sigma_opt = d[opt_var_num]
            opt_var_num += 1
            nu_opt = d[opt_var_num]
            opt_var_num += 1
            if len(init_vals_comp_var_arr[i]) > 2:
                epsilon = d[opt_var_num]
                opt_var_num += 1
            else:
                epsilon = 0

            # solve objective function
            coeff = scipy.special.gamma((nu_opt + 1) / 2) / (scipy.special.gamma(nu_opt/2) * np.sqrt(np.pi * (nu_opt - 2) * (varis[i] + sigma_opt**2)))

            factor2 = 1 + ((dists[i] + epsilon)**2) / ((varis[i] + sigma_opt**2) * (nu_opt-2))

            f_t = coeff * factor2**(-1*(nu_opt+1)/2)

            log_Li = np.log(f_t)
            log_likelihood = log_likelihood + np.nansum(log_Li)

        return -1*log_likelihood
    
    # Run minimize scalar for each parameter set
    max_l_for_us = []
    sigma_sqr_terms = []
    nu_terms = []
    epsilon_terms = []
    
    progress_bar = tqdm(total=num_variants, desc="Progress")
    for u in range(num_variants):
        param_set = u
        x_0 = init_vals
        if len(init_vals) > 2:
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1]),tuple(bnds[2])])
            epsilon_terms.append(res.x[2])
        else:    
            res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])])
        max_l_for_us.append(-res.fun)
        sigma_sqr_terms.append(res.x[0]**2)
        nu_terms.append(res.x[1])
        progress_bar.update(1)
    progress_bar.close()

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    if len(init_vals) > 2:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,nu_terms,epsilon_terms], index = ['log_L', 'sigma_sqr', 'nu', 'epsilon']).transpose()
    else:
        all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,nu_terms], index = ['log_L', 'sigma_sqr', 'nu']).transpose()
    save_dataset(all_mle, save_here_dir + 'all_mle.csv')

    # Find parameter set that gives the max likelihood
    u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # Use this parmeter set to get the model discrep term at that parameter set
    param_set = u_mle
    x_0 = init_vals
    if len(init_vals) > 2:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1]),tuple(bnds[2])]).x #val for model discrep term
    else:
        dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])]).x #val for model discrep term
    sigma = dec_vars[0]
    sigma_sqr = sigma**2
    nu = dec_vars[1]
    column_names = ['parameter_set_num', 'variance_mle', 'nu']
    optimized_vals = [u_mle, sigma_sqr, nu]

    if len(init_vals) > 2:
        epsilon = dec_vars[2]
        column_names = column_names + ['epsilon']
        optimized_vals = optimized_vals + [epsilon]

    return optimized_vals, column_names