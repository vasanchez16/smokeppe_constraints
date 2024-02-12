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

from src.simulator import SimulatedDataset
from src.observer import Observer
from src.emulator import Emulator
from src.utils import approx_mle



def noise_model(
    domain: pd.DataFrame,
    truth,
    params: dict
):
    obs = emulator(domain, params)
    return obs


def main(args):
    start_time = time.time()
    results_dir = os.getcwd() + '/results/simulation/'
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
    """
    p = 5 # Dimension of parameter space
    d = 10 # Dimension of spatiotemporal domain
    K = 10 # Number of test parameter vectors
    truth = np.random.rand(p) # "True" parameter vector u_star
    nu = 1.5 # Simulator kernel shape
    epsilon = 0.1
    delta = 12.0 # t noise scale
    omega = 3.8 # t noise df


    """
    Simulated dataset
    """
    my_simulator = SimulatedDataset(
        pixel_resolution=d,
        parameter_resolution=K,
        num_parameters=p,
        truth=truth
    )
    my_simulator.make_response(
        nu=nu,
        proportion_nuisance=0.0
    )
    my_observer = Observer(
        my_simulator,
        noise_level=epsilon
    )

    if Path(sim_loc).is_file():
        with open(sim_loc, 'r') as f:
            sim_data = pd.read_csv(sim_loc)
    else:
        my_simulator.plot_truth(fig_dir+'sim_data.png')
        sim_data = my_simulator.true_data
        with open(sim_loc, 'w') as f:
            sim_data.to_csv(f, index=False)

    if Path(obs_loc).is_file():
        with open(obs_loc, 'r') as f:
            obs_data = pd.read_csv(obs_loc)
    else:
        my_observer.make_observations()
        my_observer.plot_observations(fig_dir+'obs_data.png')
        obs_data = my_observer.SimulatedDataset.true_data
        with open(obs_loc, 'w') as f:
            obs_data.to_csv(f, index=False)


    """
    Emulate
    """
    my_emulator = Emulator(my_simulator, nu=1.5, pixelwise=True)

    if Path(params_loc).is_file():
        with open(params_loc, 'r') as f:
            params_data = np.array(pd.read_csv(params_loc))
    else:
        params_data = np.random.rand(K, p)
        with open(params_loc, 'w') as f:
            pd.DataFrame(params_data).to_csv(f, index=False)


    """
    Predict
    """
    if Path(emu_loc).is_file():
        with open(emu_loc, 'r') as f:
            emu_data = pd.read_csv(emu_loc)
    else:
        emu_variants = [
            my_emulator.emulate_variant(
                model_variant=params_data[k]
            ) for k in range(K)
        ]
        emu_data = pd.DataFrame(
            np.vstack(emu_variants),
            columns=emu_variants[0].columns
        )
        emu_data['variant'] = np.repeat(
            list(range(K)),
            d**2
        )
        with open(emu_loc, 'w') as f:
            emu_data.to_csv(f, index=False)
        my_emulator.plot_variant(fig_dir+'emu_variant.png', params_data[0])


    """
    Discrepancy
    """
    deltas = []
    nus = []
    lls = []

    for variant in np.unique(emu_data['variant']):
        data_for_variant = emu_data[emu_data.variant == variant].reset_index(
            drop=True
        )

        z = obs_data['response']
        mu = data_for_variant['response'].subtract(obs_data['response'])
        sigma = data_for_variant['emulator_std']

        delta, nu, ll = approx_mle(z, mu, sigma)
        deltas.append(delta)
        nus.append(nu)
        lls.append(ll)

    """
    if Path(mle_loc).is_file():
        with open(mle_loc, 'r') as f:
            mle_data = pd.read_csv(mle_loc)
    else:
        with open(mle_loc, 'w') as f:
            write = csv.writer(f)
            write.writerow([u_mle])
        f.close()
    """

    plt.scatter(params_data[:, 0], lls)
    plt.title('Maximum achieved likelihoods')
    plt.xlabel('$u_1$')
    plt.savefig(fig_dir+'ll.png')
    plt.close()

    plt.scatter(params_data[:, 0], deltas)
    plt.title('Noise scale estimates')
    plt.xlabel('$u_1$')
    plt.savefig(fig_dir+'delta.png')
    plt.close()

    plt.scatter(params_data[:, 0], nus)
    plt.title('Noise degrees of freedom estimates')
    plt.xlabel('$u_1$')
    plt.savefig(fig_dir+'nu.png')
    plt.close()

    mleidx = lls.index(max(lls))
    addl_var = float(deltas[mleidx]**2 * nus[mleidx] / (nus[mleidx] - 2))


    """
    Test
    """
    least_squares_e = []
    for variant in np.unique(emu_data['variant']):
        data_for_variant = emu_data[emu_data.variant == variant].reset_index(
            drop=True
        )

        z = obs_data['response']
        mu = data_for_variant['response'].subtract(obs_data['response'])
        sigma = data_for_variant['emulator_std']

        least_squares_e.append(
            (z-mu).divide(np.power(sigma + addl_var, 0.5)).iloc[mleidx]
        )

    plt.figure(figsize=(10,10))
    plt.hist(least_squares_e, density=True, bins=200)
    plt.savefig(fig_dir+'ls_residuals.png')
    plt.close()


    """
    Infer
    """


    """
    Training time report
    """
    end_time = time.time()
    during_time = end_time - start_time
    print('run time:', float(during_time))
    print('job successful')

if __name__ == "__main__":
    main(sys.argv)
