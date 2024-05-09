import pandas as pd
import numpy as np
import json
import scipy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_implaus_thresh_conv(args):
    raise NotImplementedError

def get_implaus_thresh_t(args, num_points):
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
    
    thresh = np.sqrt(np.percentile(summed,95))

    return thresh

def get_implaus_thresh_gaussian(args):

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    obs_df = pd.read_csv(save_here_dir + 'outliers.csv')

    thresh = np.sqrt(scipy.stats.chi2.ppf(0.95, sum((~obs_df.missing) & (~obs_df.outlier))))

    return thresh


# Bootstrap

def get_implaus_thresh_t_boot_pivotal(args, distances, variances):
    """
    Use bootstrap to get the implausibility threshold in the case that one threshold is needed for each emulator variant.
    """
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    # Extract evaluation parameters
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    variant_thresholds = []
    progress_bar = tqdm(total=distances.shape[1], desc="Progress")
    for i in range(distances.shape[1]):
        dists_here = distances.iloc[:,i]
        varis_here = variances.iloc[:,i]

        adj_varis = (varis_here + float(mle_df['variance_mle'])) * ((float(mle_df['nu']) - 2) / float(mle_df['nu']))
        test_stat = dists_here.div(np.power(adj_varis, 0.5))
        test_stat = test_stat[~np.isnan(test_stat)]

        implaus_arr = []
        for i in range(50000):
            boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
            implaus = np.sqrt(np.sum(boot**2))
            implaus_arr.append(implaus)
        
        implaus_thresh = np.percentile(implaus_arr, 95)
        variant_thresholds.append(implaus_thresh)
        progress_bar.update(1)
    progress_bar.close()

    return variant_thresholds

def get_implaus_thresh_t_boot_nonpivotal(args):
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

    adj_varis = (varis + float(mle_df['variance_mle'])) * ((float(mle_df['nu']) - 2) / float(mle_df['nu']))
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = []
    for i in range(50000):
        boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
        implaus = np.sqrt(np.sum(boot**2))
        implaus_arr.append(implaus)
    
    implaus_thresh = np.percentile(implaus_arr, 95)

    return implaus_thresh

def get_implaus_thresh_gauss_boot(args):
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

    adj_varis = varis + float(mle_df['variance_mle'])
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = []
    for i in range(50000):
        boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
        implaus = np.sqrt(np.sum(boot**2))
        implaus_arr.append(implaus)
    
    implaus_thresh = np.percentile(implaus_arr, 95)

    return implaus_thresh


# Functions above are currently being used, code below is in development to make code more efficient
def setup_boot(distances, variances, stats_dist_method, save_here_dir):
    """
    Doc this
    """

    if "_nonpivotal" in stats_dist_method:
        data = pd.read_csv(save_here_dir + 'mostPlausibleDistsVaris.csv')
        dists = data['dists']
        varis = data['varis']
        thresh = boot_this(dists, varis, save_here_dir)
        return thresh
    else:
        all_thresh = execute_boots(distances, variances)
        return all_thresh

def execute_boots(distances, variances, save_here_dir):
    """
    Doc this
    """

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(boot_this, distances.iloc[:,i], variances.iloc[:,i], save_here_dir) for i in range(distances.shape[1])]
        thresh_arr = [future.result() for future in futures]

    return thresh_arr

def boot_this(dists, varis, save_here_dir):
    """
    Doc this
    """
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    adj_varis = (varis + float(mle_df['variance_mle'])) * ((float(mle_df['nu']) - 2) / float(mle_df['nu']))
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = []
    for i in range(50000):
        boot = np.random.choice(test_stat, size = len(test_stat), replace=True)
        implaus = np.sqrt(np.sum(boot**2))
        implaus_arr.append(implaus)
    
    implaus_thresh = np.percentile(implaus_arr, 95)
    return implaus_thresh