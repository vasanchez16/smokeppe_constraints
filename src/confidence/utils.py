import pandas as pd
import numpy as np
import json
import scipy
import scipy.stats


def get_implaus_thresh_conv(args, conf_lvl):
    """Implausibility threshold via the convolution method. Not yet implemented."""
    raise NotImplementedError


def get_implaus_thresh_t(args, num_points, conf_lvl):
    """
    Derive the implausibility threshold using a simulated student-t distribution.

    Draws 10 000 samples from a t distribution with the MLE-estimated degrees of
    freedom, sums the squared samples across num_points replicates, and returns the
    sqrt of the conf_lvl-th percentile of that sum.

    Arguments:
    - args: argparse.Namespace with input_file and output_dir
    - num_points: int, number of non-missing observation points
    - conf_lvl: float, confidence level in percent (e.g. 95)
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    nu_opt = float(mle_df['nu'].values[0])

    sum_this = [np.random.standard_t(nu_opt, 10000) for _ in range(num_points)]
    sum_this = [i * i for i in sum_this]
    summed = np.sum(sum_this, axis=0)

    return np.sqrt(np.percentile(summed, conf_lvl))


def get_implaus_thresh_gaussian(args, conf_lvl):
    """
    Derive the implausibility threshold using the chi-squared distribution (Gaussian case).

    The threshold is the square root of the chi-squared quantile at conf_lvl, with
    degrees of freedom equal to the number of non-missing, non-outlier observations.

    Arguments:
    - args: argparse.Namespace with input_file and output_dir
    - conf_lvl: float, confidence level in percent (e.g. 95)
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    obs_df = pd.read_csv(save_here_dir + 'outliers.csv')
    dof = sum((~obs_df.missing) & (~obs_df.outlier))

    return np.sqrt(scipy.stats.chi2.ppf(conf_lvl / 100, dof))


def get_implaus_thresh_t_boot(args, conf_lvl):
    """
    Bootstrap the implausibility threshold for the student-t distribution.

    Resamples the standardised test statistics (d / sqrt(v)) from the max-likelihood
    variant 100 000 times and returns the conf_lvl-th percentile of the bootstrap
    implausibility distribution.

    Arguments:
    - args: argparse.Namespace with input_file and output_dir
    - conf_lvl: float, confidence level in percent (e.g. 95)
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    best_dists_varis = pd.read_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv')
    mle_df = pd.read_csv(save_here_dir + 'mle.csv')

    dists = best_dists_varis['dists']
    varis = best_dists_varis['varis']

    if 'epsilon' in mle_df.columns:
        dists = dists - float(mle_df['epsilon'].values[0])

    adj_varis = varis + float(mle_df['variance_mle'].values[0])
    test_stat = dists.div(np.power(adj_varis, 0.5))
    test_stat = test_stat[~np.isnan(test_stat)]

    implaus_arr = [
        np.sqrt(np.sum(np.random.choice(test_stat, size=len(test_stat), replace=True) ** 2))
        for _ in range(100000)
    ]

    return np.percentile(implaus_arr, conf_lvl)


def get_implaus_thresh_gauss_boot(args, conf_lvl):
    """
    Bootstrap the implausibility threshold for the Gaussian distribution.

    Identical procedure to get_implaus_thresh_t_boot but uses the Gaussian
    standardisation (epsilon added rather than subtracted).

    Arguments:
    - args: argparse.Namespace with input_file and output_dir
    - conf_lvl: float, confidence level in percent (e.g. 95)
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

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

    implaus_arr = [
        np.sqrt(np.sum(np.random.choice(test_stat, size=len(test_stat), replace=True) ** 2))
        for _ in range(100000)
    ]

    return np.percentile(implaus_arr, conf_lvl)
