import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import cartopy
from scipy.optimize import minimize
import sys
import os
import math
from matplotlib import ticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from pathlib import Path
from scipy.special import gamma
import argparse
import configparser
import time
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import EmulatorEval, ModelDiscrepancy, MLE
from src.emulator.simulator import Px, Pyx, sample
from src.inference.mle import Likelihood, approx_mle, psi, psi_dot_dot, p_pdf, q_pdf

# Config
config = configparser.ConfigParser()
config.read('config.ini')

input_dir = config.get('DEFAULT', 'InputDir')


def main(args):
    """
    Main function to run the simulation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    start_time = time.time()
    results_dir = args.results
    fig_dir = results_dir + 'fig/'
    sim_loc = results_dir + 'sim_data.csv'
    obs_loc = results_dir + 'obs_data.csv'
    params_loc = results_dir + 'params_data.csv'
    emu_loc = results_dir + 'emu_data.csv'
    mle_loc = results_dir + 'mle_data.csv'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    EmulatorEval(args)
    ModelDiscrepancy(args)
    MLE(args)


    """
    Training time report
    """
    end_time = time.time()
    during_time = end_time - start_time
    print('run time:', float(during_time))
    print('job successful')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument("--savefigs", action="store_true", default=True)
    parser.add_argument(
        "--results",
        type=str,
        default=input_dir
    )
    args = parser.parse_args()
    main(args)
