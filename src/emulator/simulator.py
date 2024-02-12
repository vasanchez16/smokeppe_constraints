# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:24:34 2022

@author: james

Instances of this class represent the simulator or just the data we take for
example from it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class SimulatedDataset:

    def __init__(
            self,
            pixel_resolution=10,
            parameter_resolution=10,
            num_parameters=1,
            truth=[0.33]
        ):
        """
        On initialization, we fix some parameters for the problem type and
        then methodically simulate the data.

        Parameters
        ----------
        pixel_resolution : integer, optional
            Number of latitude (resp. longitude) coordinates. The default is
            10.
        parameter_resolution : integer, optional
            Number of parameter values (i.e. model variants) to include in the
            simulated dataset. The default is 20.
        truth : float, optional
            The "true" model variant. The default is 0.33.

        Returns
        -------
        None.

        """
        
        if (pd.DataFrame(truth).shape[0] != num_parameters):
            raise ValueError("The true variant does not match the dimension "+
                             "of the input space.")
            return
        
        # Create the spatial mesh for the map
        self.pixel_res = np.linspace(0, 1, pixel_resolution)
        self.pixels = pd.DataFrame(
            data=[[lat, lon]
                  for lat in self.pixel_res
                  for lon in self.pixel_res],
            columns=['latitude',
                     'longitude']
        )
        
        # Fix a collection of model variants to obtain
        self.num_parameters = num_parameters

        self.parameter_levels = pd.DataFrame(
            np.random.uniform(
                low=0, high=1,
                size=(parameter_resolution, num_parameters)
        ))

        self.truth = pd.DataFrame(truth)

        # Combine the map data with the model variants to tabulate all of the
        # data we need to define the full simulated dataset
        param_levels_to_input = pd.concat(
            [self.parameter_levels,
             self.truth.transpose()],
            axis=0
        ).to_numpy().tolist()

        self.inputs = pd.DataFrame(
            data=[[lat, lon, *p]
                  for lat in self.pixel_res
                  for lon in self.pixel_res
                  for p in param_levels_to_input],
            columns=['latitude',
                     'longitude',
                     *['parameter' + str(digit)
                       for digit in range(num_parameters)]]
        )

        return


    def make_response(
            self,
            nu=1.5,
            random_seed=0,
            random_state=1,
            proportion_nuisance=0.75,
            length_scale_bounds=(0.01, 2.0),
            kernel_params=None
        ):
        """
        Produce a random Gaussian Process and let a realization of the process
        be the surface which implicitly defines the simulator.

        Parameters
        ----------
        random_seed : integer, optional
            Seed for drawing kernel parameters to decide variable importance.
            The default is 0.
        random_state : integer, optional
            Seed for drawing a realization from the Gaussian Process
            representing the simulator. The default is 0.
        proportion_nuisance : float
            The proportion of self.num_parameters which are virtually
            unimportant to the response
        length_scale_bounds : tuple of floats, optional
            This range decides the distances at which points become
            uncorrelated in the domain. The kernel parameters will be drawn
            from this range. The default is (0.01, 2.0).

        Returns
        -------
        None.

        """

        # Decide how many of the parameters are not *very* influential on the
        # response (i.e. will have very large length scales).
        np.random.seed(random_seed)
        num_nuisance = int(proportion_nuisance*(self.inputs.shape[1] - 2))

        # Fix the length scales from which we produce the true surface.
        #
        # The uninfluential length scales will fall between 50 and 100.
        if kernel_params is None:
            self.kernel_params = np.concatenate((
                np.random.uniform(
                    length_scale_bounds[0],
                    length_scale_bounds[1],
                    size=self.inputs.shape[1] - num_nuisance
                ),
                np.random.uniform(
                    1000,
                    1100,
                    size=num_nuisance
                )
            ))
        else:
            self.kernel_params = kernel_params

        # Define the Gaussian Process a realization of which is used to define
        # the "true" simulation output.
        self.kernel = Matern(
            length_scale=self.kernel_params,
            nu=nu
        )
        self.simulator = GaussianProcessRegressor(
            kernel=self.kernel
        )

        # Draw a realization, defining the "true" response.
        response = pd.DataFrame(
            self.simulator.sample_y(self.inputs.values,
                                    n_samples=1,
                                    random_state=random_state)
        )

        # We first manually convert the inputs table into rows with which we
        # can associate the response values.
        true_rows = [
            np.array_equal(
                self.inputs.iloc[:, 2:].to_numpy().tolist()[k],
                np.array(self.truth.transpose())[0]
            ) 
            for k in range(self.inputs.shape[0])
        ]

        # Build the "true" data set, rows evaluated at the true model variant.
        true_data = self.inputs.loc[
            true_rows, ].copy()
        true_data.loc[:, 'response'] = response[
            true_rows
        ]
        self.true_data = true_data.reset_index(drop=True)

        # Build the "simulated" data set, rows evaluated at other variants.
        sim_data = self.inputs.loc[
            [not row for row in true_rows], ].copy()
        sim_data.loc[:, 'response'] = response[
            [not row for row in true_rows]
        ]
        self.sim_data = sim_data.reset_index(drop=True)

        self.is_simulated = True

        return


    def plot_truth(self, fig_loc):
        """
        Draw what the true response looks like according to the hypothetical
        simulator and the parameter level which was prescribed as true on
        initialization.

        Returns
        -------
        None.

        """
        sc = plt.scatter(
            self.true_data.longitude,
            self.true_data.latitude,
            c=self.true_data.response
        )
        plt.colorbar(sc)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Simulated data at the true model variant')
        plt.savefig(fig_loc)
        plt.close()
        return