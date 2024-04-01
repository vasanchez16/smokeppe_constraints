import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import math
import scipy
from scipy.special import gamma
import scipy.stats
import cartopy
from scipy.optimize import minimize_scalar, minimize
import sys
import os
from matplotlib import ticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import json
from src.inference.utils import save_dataset
sys.path.append(os.getcwd())


class Likelihood:
    def __init__(self, P_x, P_y_x):
        self.params = P_x.params
        self.P_x = P_x
        self.P_y_x = P_y_x


    def q_pdf(self, y, x):
        y_hat = self.y_hat(y, x)
        C = np.sqrt(-2*math.pi/self.psi_dot_dot(y_hat, x))
        return C*np.exp(self.psi(y_hat, x))


    def y_hat(self, y, x):
        roots, psi_roots = None, None
        roots = list(np.real(self.c_psi_dot(y, x).roots()))
        psi_roots = [self.psi(root, x) for root in roots]
        return roots[min([2, np.argmax(psi_roots)])]


    def psi(self, y, x):
        return self.P_x.logpdf(x) + self.P_y_x.logpdf(y)


    def c_psi_dot(self, y, x):
        mu, sigma = self.params['mu'], self.params['sigma']
        delta, nu = self.params['delta'], self.params['nu']
        coeffs = [
            -mu*(y**2 + delta**2*nu) - y*(nu+1)*sigma**2,
            2*y*mu + y**2 + delta**2*nu + (nu+1)*sigma**2,
            -2*y - mu,
            1
        ]
        return Polynomial(coeffs)


    def psi_dot_dot(self, y, x):
        T1 = -1/self.params['sigma']**2
        T2 = -(self.params['nu']+1)*\
            (
                self.params['nu']*self.params['delta']**2 - (y - np.inner(x, self.params['beta']))**2
            )/\
            (
                self.params['nu']*self.params['delta']**2 + (y - np.inner(x, self.params['beta']))**2
            )
        return T1 + T2


def psi(y, x, mu, sigma, nu, delta, beta):
    T1 = -0.5*np.log(2*np.pi*sigma**2) - 0.5*(y - mu)**2/sigma**2
    T2 = -0.5*np.log(np.pi*nu*delta**2) + \
        np.log(gamma((nu+1)/2) / gamma(nu/2)) - \
            (nu+1)*(1+(y-x)**2/(nu*delta**2))/2
    return T1 + T2


def psi_dot_dot(y, x, mu, sigma, nu, delta, beta):
    T1 = -1/sigma**2
    T2 = -(nu+1)*(nu*delta**2 - (y-x)**2)/\
        (nu*delta**2 - (y-x)**2)
    return T1 + T2


def p_pdf(y, mu, sigma, nu, delta, beta):
    p = []
    for j in range(len(y)):
        def f(x):
            return np.exp(psi(y, x, mu, sigma, nu, delta, beta))
        res, err = quad_vec(f, -10, 10)
        p.append(res)
    return np.array(p).reshape(-1)


def q_pdf(y, x, l, mu, sigma, nu, delta, beta):
    p = []
    for j in range(len(y)):
        y_hat = l.y_hat(y[j], x[j])
        C = np.sqrt(
            -2*math.pi/psi_dot_dot(
                y_hat, x[j], mu, sigma, nu, delta, beta
            )
        )
        p.append(C*np.exp(psi(
            y_hat, x[j], mu, sigma, nu, delta, beta
        )))
    return np.array(p).reshape(-1)


def approx_mle(y, x, l, theta1=[0., 1.], theta2=[10, 10], theta3=3., laplace=True):
    """
    Value:
        delta, nu, log-likelihood
    """
    if laplace:
        pdf = q_pdf
    else:
        pdf = p_pdf
    log_lik = minimize(
        lambda theta0: -np.sum(np.log(pdf(
            y, x, l,
            *theta1, theta0[0], theta0[1], theta3
        ))),
        theta2,
        method='L-BFGS-B',
        tol=10e-10,
        bounds=[(2.5, np.inf), (0.1, np.inf)]
    )
    opt_vals = [log_lik.x[0], log_lik.x[1], -log_lik.fun]
    column_names = ['x_0','x_1','fun_val']
    return opt_vals, column_names

def mle_t(args, distances, variances, num_variants):
    """
    mle analysis using the student t approximation
    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    bnds = eval_params['MLE_optimization']['bounds']
    init_vals = eval_params['MLE_optimization']['initial_vals']

    def minus_log_l(d):
        # get dists for one param set
        dists = distances.iloc[:,param_set]
        varis = variances.iloc[:,param_set]
        # Log likelihood, to be maximized
        sigma_opt = d[0]
        nu_opt = d[1]

        coeff = scipy.special.gamma((nu_opt + 1) / 2) / (scipy.special.gamma(nu_opt/2) * np.sqrt(np.pi * (nu_opt - 2) * (varis + sigma_opt**2)))

        factor2 = 1 + (dists**2) / ((varis + sigma_opt**2) * (nu_opt-2))

        f_t = coeff * factor2**(-1*(nu_opt+1)/2)

        log_Li = np.log(f_t)
        log_likelihood = np.nansum(log_Li)
        return -1*log_likelihood
    
    # Run minimize scalar for each parameter set
    max_l_for_us = []
    sigma_sqr_terms = []
    nu_terms = []
    for u in range(num_variants):
        param_set = u
        if u%1000 == 0:
            print(f'Parameter set: {u}')
        x_0 = init_vals
        res = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])])
        max_l_for_us.append(-res.fun)
        sigma_sqr_terms.append(res.x[0]**2)
        nu_terms.append(res.x[1])

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'

    all_mle = pd.DataFrame([max_l_for_us,sigma_sqr_terms,nu_terms], index = ['log_L', 'sigma_sqr', 'nu']).transpose()
    save_dataset(all_mle, save_here_dir + 'all_mle.csv')

    # Find parameter set that gives the max likelihood
    u_mle = max_l_for_us.index(max(max_l_for_us)) # param combination number
    # Use this parmeter set to get the model discrep term at that parameter set
    param_set = u_mle
    x_0 = init_vals
    dec_vars = minimize(minus_log_l,x_0,bounds=[tuple(bnds[0]),tuple(bnds[1])]).x #val for model discrep term
    sigma = dec_vars[0]
    sigma_sqr = sigma**2
    nu = dec_vars[1]
    column_names = ['parameter_set_num', 'variance_mle', 'nu']
    optimized_vals = [u_mle, sigma_sqr, nu]

    return optimized_vals, column_names