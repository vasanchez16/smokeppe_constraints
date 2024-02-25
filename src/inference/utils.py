"""
For tasks such as the Laplace approximation step
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import gamma
import scipy.stats
import cartopy
from scipy.optimize import minimize
import sys
import os
from matplotlib import ticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import json
sys.path.append(os.getcwd())


def sigma_u(u, variances):
    return variances.iloc[:, u]

def mu_u(u, z, distances):
    z_mu = distances.iloc[:, u]
    return z - z_mu

def psi(x, z_i, delta, nu, mu_i, sigma_i):
    log_p_x_u = -0.5*np.log(2*np.pi*sigma_i**2) - 0.5*(x - mu_i)**2/sigma_i**2
    log_p_z_x = -0.5*np.log(np.pi*nu*delta**2) + \
        np.log(gamma((nu+1)/2) / gamma(nu/2)) - \
            (nu+1)*(1+(z_i-x)**2/(nu*delta**2))/2
    return log_p_x_u + log_p_z_x

def psi_dot_dot(x, z_i, delta, nu, mu_i, sigma_i):
    log_p_x_u_dot_dot = -1/sigma_i**2
    log_p_z_x_dot_dot = -(nu+1)*(nu*delta**2 - (x-z_i)**2)/\
        (nu*delta**2 - (x-z_i)**2)
    return log_p_x_u_dot_dot + log_p_z_x_dot_dot

def compute_x_hat(z_i, delta, nu, mu_i, sigma_i):
    if sum(np.isnan([z_i, mu_i, sigma_i])) > 1:
        return np.nan

    r = np.polynomial.polynomial.Polynomial(
        (
            -mu_i*(z_i**2 + delta**2*nu) - z_i*(nu+1)*sigma_i,
            2*z_i*mu_i + z_i**2 + delta**2*nu + (nu+1)*sigma_i,
            -2*z_i - mu_i,
            1
        )
    )
    roots = np.real(r.roots())

    x_hat = roots[0]
    for root in roots:
        if psi(root, z_i, delta, nu, mu_i, sigma_i) > \
            psi(x_hat, z_i, delta, nu, mu_i, sigma_i):
            x_hat = root
    return x_hat

def log_q_i(z_i, delta, nu, mu_i, sigma_i):
    x_hat = compute_x_hat(z_i, delta, nu, mu_i, sigma_i)
    return 0.5*np.log(
        2*np.pi / -psi_dot_dot(x_hat, z_i, delta, nu, mu_i, sigma_i)
    ) + psi(x_hat, z_i, delta, nu, mu_i, sigma_i)

def log_q(z, delta, nu, mu, sigma):
    # assert len(z) == len(mu) and len(z) == len(sigma)
    return np.nansum([
        log_q_i(z[i], delta, nu, mu[i], sigma[i]) for i in range(len(z))
    ])

def approx_mle(z, mu, sigma, theta0=[10, 10]):
    """
    Value:
        delta, nu, log-likelihood
    """
    log_lik = minimize(
        lambda theta: -log_q(z, theta[0], theta[1], mu, sigma),
        theta0,
        method='L-BFGS-B',
        bounds=[(0.1, np.inf), (2.5, np.inf)]
    )
    return log_lik.x[0], log_lik.x[1], -log_lik.fun

def save_dataset(data, save_path):
    """
    Arguments:
    data: pandas DataFrame Obj
    Data to be saved
    save_path: str
    Path where this data will be saved
    """
    data.to_csv(save_path)
    return

def save_indexed_dataset():
    """
    Save distances separately for specific parameter set.
    Implement later if needed
    """
    return

def set_up_directories(args):
    
    eval_params = args.input_file
    with open(eval_params,'r') as file:
        evaluation_parameters = json.load(file)
    run_label = evaluation_parameters['run_label']

    if not os.path.exists(args.output_dir + run_label):
        os.mkdir(args.output_dir + run_label)
    return