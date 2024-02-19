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

from EmulatorEval import EmulatorEval
from ModelDiscrepancy import ModelDiscrepancy
from MLE import MLE
from src.utils import set_up_directories

# Config
config = configparser.ConfigParser()
config.read('config.ini')

input_file = config.get('DEFAULT', 'InputFile')
output_dir = config.get('DEFAULT', 'OutputDir')


def main(args):
    """
    Main function to run the simulation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    start_time = time.time()


    """
    Set up directories
    """
    set_up_directories(args)


    """
    Run pipelines
    """
    EmulatorEval(args)
    ModelDiscrepancy(args)
    MLE(args)
    # Implausibilities(args)
    # FreqConfSet(args)


    """
    Runtime report
    """
    end_time = time.time()
    during_time = end_time - start_time
    print('run time:', float(during_time))
    print('job successful')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument("--savefigs", action="store_true", default=False)
    parser.add_argument(
        "--input_file",
        type=str,
        default=input_file
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir
    )
    args = parser.parse_args()
    main(args)


# Formerly toy_simulation_laplace

# import pandas as pd
# import numpy as np
# from numpy.polynomial.polynomial import Polynomial
# import matplotlib.pyplot as plt
# import scipy
# import scipy.stats
# import cartopy
# from scipy.optimize import minimize
# import sys
# import os
# import math
# from matplotlib import ticker
# import cartopy.crs as ccrs
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import time
# from pathlib import Path
# from scipy.special import gamma
# import argparse
# import configparser
# import time
# import sys
# import os

# # Add parent directory to sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(parent_dir)

# from src.emulator.simulator import Px, Pyx, sample
# from src.inference.mle import Likelihood, approx_mle, psi, psi_dot_dot, p_pdf, q_pdf

# # Config
# config = configparser.ConfigParser()
# config.read('config.ini')

# input_dir = config.get('DEFAULT', 'InputDir')


# def main(args):
#     """
#     Main function to run the simulation.

#     Args:
#         args (argparse.Namespace): Command-line arguments.

#     Returns:
#         None
#     """
#     start_time = time.time()
#     results_dir = args.results
#     fig_dir = results_dir + 'fig/'
#     sim_loc = results_dir + 'sim_data.csv'
#     obs_loc = results_dir + 'obs_data.csv'
#     params_loc = results_dir + 'params_data.csv'
#     emu_loc = results_dir + 'emu_data.csv'
#     mle_loc = results_dir + 'mle_data.csv'

#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#     if not os.path.exists(fig_dir):
#         os.makedirs(fig_dir)


#     """
#     Problem set-up

#     Fix (mu, sigma). For various student noise settings (nu, delta^2), we 
#     estimate the MLE for (mu, sigma, nu, delta^2) by LBFGS-B on the weighted
#     MSE.
#     """
#     n = 10
#     n_sim = 2
#     range_nu = np.linspace(2., 20., 2)
#     range_delta = np.linspace(10e-2, 20, 2)
#     params_list = [{
#         'mu': 0.,
#         'sigma': 1.,
#         'nu': nu,
#         'delta': delta,
#         'beta': 3.
#     } for nu in range_nu for delta in range_delta]


#     """
#     Experiment
#     """
#     for params in params_list:
#         nus = []
#         deltas = []
#         lls = []

#         P1 = Px(params)
#         for sim in range(n_sim):
#             X, Y = sample(n, params, sim)
#             P2 = Pyx(X, params)
#             l = Likelihood(P1, P2)

#             nu, delta, ll = approx_mle(Y, X, l, theta2=[params['nu'], params['delta']])
#             nus.append(nu)
#             deltas.append(delta)
#             lls.append(ll)
    
#         plt.scatter(nus, deltas, alpha=0.5, label='Estimates')
#         plt.scatter(params['nu'], params['delta'], label='Truth')
#         plt.legend()
#         plt.title('Noise parameter MLEs ($(\\nu,\delta)=$({:.2f},{:.2f}))'.format(params['nu'], params['delta']))
#         plt.xlabel('$\\nu$')
#         plt.ylabel('$\delta$')
#         plt.savefig(fig_dir+'mle_{:.2f}_{:.2f}.png'.format(params['nu'], params['delta']))
#         plt.close()

#         # Plot likelihood
#         # Create a pandas DataFrame
#         df = pd.DataFrame(columns=['nu', 'delta', 'likelihood'])

#         # Iterate over the values in range_nu and range_delta
#         for nu in range_nu:
#             for delta in range_delta:
#                 # Append a row to the DataFrame with the current values of nu and delta
#                 likelihood = np.prod(q_pdf(Y, X, l, params['mu'], params['sigma'], params['nu'], params['delta'], params['beta']))
#                 df = df.append({'nu': nu, 'delta': delta, 'likelihood': likelihood}, ignore_index=True)

#         # Plot df
#         plt.scatter(df['nu'], df['delta'], c=np.log(df['likelihood']), cmap='viridis')
#         plt.xlabel('nu')
#         plt.ylabel('delta')
#         plt.colorbar()
#         plt.savefig(fig_dir + 'likelihood_surface_{:.2f}_{:.2f}.png'.format(params['nu'], params['delta']))
#         plt.close()


#     """
#     Training time report
#     """
#     end_time = time.time()
#     during_time = end_time - start_time
#     print('run time:', float(during_time))
#     print('job successful')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run simulation.")
#     parser.add_argument("--savefigs", action="store_true", default=True)
#     parser.add_argument(
#         "--results",
#         type=str,
#         default=input_dir
#     )
#     args = parser.parse_args()
#     main(args)


# Formerly emulator_simulation

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scipy.stats
# import cartopy
# from scipy.optimize import minimize
# import sys
# import os
# from matplotlib import ticker
# import cartopy.crs as ccrs
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import time
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))

# from src.simulator import SimulatedDataset
# from src.observer import Observer
# from src.emulator import Emulator
# from src.utils import approx_mle



# def noise_model(
#     domain: pd.DataFrame,
#     truth,
#     params: dict
# ):
#     obs = emulator(domain, params)
#     return obs


# def main(args):
#     start_time = time.time()
#     results_dir = os.getcwd() + '/results/simulation/'
#     fig_dir = results_dir + 'fig/'
#     sim_loc = results_dir + 'sim_data.csv'
#     obs_loc = results_dir + 'obs_data.csv'
#     params_loc = results_dir + 'params_data.csv'
#     emu_loc = results_dir + 'emu_data.csv'
#     mle_loc = results_dir + 'mle_data.csv'

#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#     if not os.path.exists(fig_dir):
#         os.makedirs(fig_dir)


#     """
#     Problem set-up
#     """
#     p = 5 # Dimension of parameter space
#     d = 10 # Dimension of spatiotemporal domain
#     K = 10 # Number of test parameter vectors
#     truth = np.random.rand(p) # "True" parameter vector u_star
#     nu = 1.5 # Simulator kernel shape
#     epsilon = 0.1
#     delta = 12.0 # t noise scale
#     omega = 3.8 # t noise df


#     """
#     Simulated dataset
#     """
#     my_simulator = SimulatedDataset(
#         pixel_resolution=d,
#         parameter_resolution=K,
#         num_parameters=p,
#         truth=truth
#     )
#     my_simulator.make_response(
#         nu=nu,
#         proportion_nuisance=0.0
#     )
#     my_observer = Observer(
#         my_simulator,
#         noise_level=epsilon
#     )

#     if Path(sim_loc).is_file():
#         with open(sim_loc, 'r') as f:
#             sim_data = pd.read_csv(sim_loc)
#     else:
#         my_simulator.plot_truth(fig_dir+'sim_data.png')
#         sim_data = my_simulator.true_data
#         with open(sim_loc, 'w') as f:
#             sim_data.to_csv(f, index=False)

#     if Path(obs_loc).is_file():
#         with open(obs_loc, 'r') as f:
#             obs_data = pd.read_csv(obs_loc)
#     else:
#         my_observer.make_observations()
#         my_observer.plot_observations(fig_dir+'obs_data.png')
#         obs_data = my_observer.SimulatedDataset.true_data
#         with open(obs_loc, 'w') as f:
#             obs_data.to_csv(f, index=False)


#     """
#     Emulate
#     """
#     my_emulator = Emulator(my_simulator, nu=1.5, pixelwise=True)

#     if Path(params_loc).is_file():
#         with open(params_loc, 'r') as f:
#             params_data = np.array(pd.read_csv(params_loc))
#     else:
#         params_data = np.random.rand(K, p)
#         with open(params_loc, 'w') as f:
#             pd.DataFrame(params_data).to_csv(f, index=False)


#     """
#     Predict
#     """
#     if Path(emu_loc).is_file():
#         with open(emu_loc, 'r') as f:
#             emu_data = pd.read_csv(emu_loc)
#     else:
#         emu_variants = [
#             my_emulator.emulate_variant(
#                 model_variant=params_data[k]
#             ) for k in range(K)
#         ]
#         emu_data = pd.DataFrame(
#             np.vstack(emu_variants),
#             columns=emu_variants[0].columns
#         )
#         emu_data['variant'] = np.repeat(
#             list(range(K)),
#             d**2
#         )
#         with open(emu_loc, 'w') as f:
#             emu_data.to_csv(f, index=False)
#         my_emulator.plot_variant(fig_dir+'emu_variant.png', params_data[0])


#     """
#     Discrepancy
#     """
#     deltas = []
#     nus = []
#     lls = []

#     for variant in np.unique(emu_data['variant']):
#         data_for_variant = emu_data[emu_data.variant == variant].reset_index(
#             drop=True
#         )

#         z = obs_data['response']
#         mu = data_for_variant['response'].subtract(obs_data['response'])
#         sigma = data_for_variant['emulator_std']

#         delta, nu, ll = approx_mle(z, mu, sigma)
#         deltas.append(delta)
#         nus.append(nu)
#         lls.append(ll)

#     """
#     if Path(mle_loc).is_file():
#         with open(mle_loc, 'r') as f:
#             mle_data = pd.read_csv(mle_loc)
#     else:
#         with open(mle_loc, 'w') as f:
#             write = csv.writer(f)
#             write.writerow([u_mle])
#         f.close()
#     """

#     plt.scatter(params_data[:, 0], lls)
#     plt.title('Maximum achieved likelihoods')
#     plt.xlabel('$u_1$')
#     plt.savefig(fig_dir+'ll.png')
#     plt.close()

#     plt.scatter(params_data[:, 0], deltas)
#     plt.title('Noise scale estimates')
#     plt.xlabel('$u_1$')
#     plt.savefig(fig_dir+'delta.png')
#     plt.close()

#     plt.scatter(params_data[:, 0], nus)
#     plt.title('Noise degrees of freedom estimates')
#     plt.xlabel('$u_1$')
#     plt.savefig(fig_dir+'nu.png')
#     plt.close()

#     mleidx = lls.index(max(lls))
#     addl_var = float(deltas[mleidx]**2 * nus[mleidx] / (nus[mleidx] - 2))


#     """
#     Test
#     """
#     least_squares_e = []
#     for variant in np.unique(emu_data['variant']):
#         data_for_variant = emu_data[emu_data.variant == variant].reset_index(
#             drop=True
#         )

#         z = obs_data['response']
#         mu = data_for_variant['response'].subtract(obs_data['response'])
#         sigma = data_for_variant['emulator_std']

#         least_squares_e.append(
#             (z-mu).divide(np.power(sigma + addl_var, 0.5)).iloc[mleidx]
#         )

#     plt.figure(figsize=(10,10))
#     plt.hist(least_squares_e, density=True, bins=200)
#     plt.savefig(fig_dir+'ls_residuals.png')
#     plt.close()


#     """
#     Infer
#     """


#     """
#     Training time report
#     """
#     end_time = time.time()
#     during_time = end_time - start_time
#     print('run time:', float(during_time))
#     print('job successful')

# if __name__ == "__main__":
#     main(sys.argv)
