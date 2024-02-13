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

    return log_lik.x[0], log_lik.x[1], -log_lik.fun
