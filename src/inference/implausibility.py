import pandas as pd
import numpy as np
import json

def get_implaus_thresh_conv(args):
    return

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
    nu_opt = float(mle_df.loc[0,'nu_opt'])

    sum_this = []
    for i in range(num_points):
        test_t = np.random.standard_t(nu_opt,10000)
        sum_this.append(test_t)
    sum_this = [i * i for i in sum_this]
    summed = np.sum(sum_this,axis=0)
    
    thresh = np.sqrt(np.percentile(summed,95))

    return thresh

def get_implaus_thresh_gaussian(args):
    return