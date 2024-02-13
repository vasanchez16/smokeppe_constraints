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


    """
    Problem set-up

    Fix (mu, sigma). For various student noise settings (nu, delta^2), we 
    estimate the MLE for (mu, sigma, nu, delta^2) by LBFGS-B on the weighted
    MSE.
    """
    n = 10
    n_sim = 2
    range_nu = np.linspace(2., 20., 2)
    range_delta = np.linspace(10e-2, 20, 2)
    params_list = [{
        'mu': 0.,
        'sigma': 1.,
        'nu': nu,
        'delta': delta,
        'beta': 3.
    } for nu in range_nu for delta in range_delta]


    """
    Experiment
    """
    for params in params_list:
        nus = []
        deltas = []
        lls = []

        P1 = Px(params)
        for sim in range(n_sim):
            X, Y = sample(n, params, sim)
            P2 = Pyx(X, params)
            l = Likelihood(P1, P2)

            nu, delta, ll = approx_mle(Y, X, l, theta2=[params['nu'], params['delta']])
            nus.append(nu)
            deltas.append(delta)
            lls.append(ll)
    
        plt.scatter(nus, deltas, alpha=0.5, label='Estimates')
        plt.scatter(params['nu'], params['delta'], label='Truth')
        plt.legend()
        plt.title('Noise parameter MLEs ($(\\nu,\delta)=$({:.2f},{:.2f}))'.format(params['nu'], params['delta']))
        plt.xlabel('$\\nu$')
        plt.ylabel('$\delta$')
        plt.savefig(fig_dir+'mle_{:.2f}_{:.2f}.png'.format(params['nu'], params['delta']))
        plt.close()

        # Plot likelihood
        # Create a pandas DataFrame
        df = pd.DataFrame(columns=['nu', 'delta', 'likelihood'])

        # Iterate over the values in range_nu and range_delta
        for nu in range_nu:
            for delta in range_delta:
                # Append a row to the DataFrame with the current values of nu and delta
                likelihood = np.prod(q_pdf(Y, X, l, params['mu'], params['sigma'], params['nu'], params['delta'], params['beta']))
                df = df.append({'nu': nu, 'delta': delta, 'likelihood': likelihood}, ignore_index=True)

        # Plot df
        plt.scatter(df['nu'], df['delta'], c=np.log(df['likelihood']), cmap='viridis')
        plt.xlabel('nu')
        plt.ylabel('delta')
        plt.colorbar()
        plt.savefig(fig_dir + 'likelihood_surface_{:.2f}_{:.2f}.png'.format(params['nu'], params['delta']))
        plt.close()


    """
    Training time report
    """
    end_time = time.time()
    during_time = end_time - start_time
    print('run time:', float(during_time))
    print('job successful')


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
