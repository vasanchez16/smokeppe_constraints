import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import cartopy
from scipy.optimize import minimize
import sys
import os
from matplotlib import ticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.emulator.simulator import SimulatedDataset
from src.emulator.observer import Observer
from src.emulator.emulator import Emulator
from src.inference.utils import approx_mle


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


def sample(N: int, params: dict, random_state: int=1):
    X = Px(params).rvs(N, random_state)
    eps = Pyx(np.inner(X, params['beta']), params).rvs(N, random_state)
    return X, X + eps


class Px:
    def __init__(self, params):
        self.params = params
        self._dist = scipy.stats.norm(params['mu'], params['sigma'])
    def rvs(self, N, random_state):
        return self._dist.rvs(N, random_state=random_state)
    def pdf(self, x):
        return self._dist.pdf(x)
    def logpdf(self, x):
        return self._dist.logpdf(x)


class Pyx:
    """
    The probability density above is defined in the “standardized” form. To shift
    and/or scale the distribution use the loc and scale parameters. Specifically,
    t.pdf(x, df, loc, scale) is identically equivalent to t.pdf(y, df) / scale
    with y = (x - loc) / scale. Note that shifting the location of a distribution
    does not make it a “noncentral” distribution; noncentral generalizations of
    some distributions are available in separate classes.
    """
    def __init__(self, x, params):
        self.params = params
        self._dist = scipy.stats.t(params['nu'], np.inner(x, params['beta']), params['delta']**2)
    def rvs(self, N, random_state):
        return self._dist.rvs(N, random_state=random_state)
    def pdf(self, y):
        return self._dist.pdf(y)
    def logpdf(self, y):
        return self._dist.logpdf(y)


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


def main(args):
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
        default="/ocean/projects/atm200005p/jcarzon/results/"
    )
    args = parser.parse_args()
    main(args)
