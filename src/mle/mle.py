import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.storage.utils import save_dataset, save_indexed_dataset
from .gauss import mle_gauss
from .student_t import mle_t
import json

from mpi4py import MPI

COMM = MPI.COMM_WORLD
CRANK = COMM.Get_rank()
CSIZE = COMM.Get_size()
CROOT = 0

def mle(args):
    """
    Collect datasets
    """
    if CRANK == CROOT:
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

    """
    Calculate MLE for model discrepancy
    """
    if stats_dist_method == 'convolution':
        raise NotImplementedError('Laplace approximation by convolution method not implemented')
    elif 'student-t' in stats_dist_method:
        opt_vals,col_names = mle_t(args)
    elif 'gaussian' in stats_dist_method:
        opt_vals,col_names = mle_gauss(args, num_variants)

    """
    Save datasets
    """
    if CRANK == CROOT:
        # Save metrics to dataframe and csv
        mle_df = pd.DataFrame(opt_vals,index=col_names).transpose()
        save_dataset(mle_df, save_here_dir + 'mle.csv')

    return
