import pandas as pd
import numpy as np
import json
import scipy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_implaus_thresh_conv(args, conf_lvl):
    raise NotImplementedError

def get_implaus_thresh_t(args, num_points, conf_lvl):
    """
    Notes, This is for the method with the t-distribution approximation. It simulates
    a t distribution and gets the threshold for confidence intervals.
    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    nu_opt = float(mle_df['nu'])

    sum_this = []
    for i in range(num_points):
        test_t = np.random.standard_t(nu_opt,10000)
        sum_this.append(test_t)
    sum_this = [i * i for i in sum_this]
    summed = np.sum(sum_this,axis=0)
    
    thresh = np.sqrt(np.percentile(summed,conf_lvl))

    return thresh

def get_implaus_thresh_gaussian(args, conf_lvl):

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    obs_df = pd.read_csv(save_here_dir + 'outliers.csv')

    thresh = np.sqrt(scipy.stats.chi2.ppf(conf_lvl / 100, sum((~obs_df.missing) & (~obs_df.outlier))))

    return thresh

# Bootstrap

def get_implaus_thresh_t_boot(args, conf_lvl):
    """
    Use bootstrap to get the implausibility thresold in the case that one threshold applies to all emulator variants.
    """
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    best_dists_varis = pd.read_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv')
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    dists = best_dists_varis['dists']
    varis = best_dists_varis['varis']

    if 'epsilon' in mle_df.columns:
        dists = dists + float(mle_df['epsilon'])

    adj_varis = varis + float(mle_df['variance_mle'])
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = []
    for i in range(100000):
        boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
        implaus = np.sqrt(np.sum(boot**2))
        implaus_arr.append(implaus)
    
    implaus_thresh = np.percentile(implaus_arr, conf_lvl)

    return implaus_thresh

def get_implaus_thresh_gauss_boot(args, conf_lvl):
    """
    Use bootstrap to get the implausibility threshold in the case that one threshold applies to all emulator variants and
    the distribution is best approximated by a Gaussian.
    """
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    best_dists_varis = pd.read_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv')
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    dists = best_dists_varis['dists']
    varis = best_dists_varis['varis']

    if 'epsilon' in mle_df.columns:
        dists = dists + float(mle_df['epsilon'])

    adj_varis = varis + float(mle_df['variance_mle'])
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = []
    for i in range(100000):
        boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
        implaus = np.sqrt(np.sum(boot**2))
        implaus_arr.append(implaus)
    
    implaus_thresh = np.percentile(implaus_arr, conf_lvl)

    return implaus_thresh
