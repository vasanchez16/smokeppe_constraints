"""
Created on Wed Jun 15 14:29:25 2022

@author: james

Instances of this class represent the climate model emulator and provide
quick runs of the emulator only with access to samples of the true response
surface.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class Emulator:

    def __init__(
        self,
        SimulatedDataset,
        nu=0.5,
        pixelwise=False,
        exclude_features=[]
    ):
        """
        Collect the parameters and data we need to define the emulator.

        Parameters
        ----------
        SimulatedDataset : SimulatedDataset
            The dataset that we need to emulate the simulator.
        nu : float, optional
            The shape parameter for the Gaussian Process model with which we
            emulate the simulator. The default is 0.5.
        pixelwise : bool, optional
            Emulate each pixel independently, or have one emulator which also
            learns latitude and longitude dependence? The default is False.
        exclude_features : list
            Features (parameters) which possibly helped to generate the
            SimulatedDataset but which are excluded from training the emulator.
            The default is the empty list.

        Returns
        -------
        None.

        """

        self.SimulatedDataset = SimulatedDataset
        self.true_data = self.SimulatedDataset.sim_data
        self.nu = nu
        self.pixelwise = pixelwise
        self.exclude_features = exclude_features
        if self.pixelwise:
            self.exclude_features += ['latitude', 'longitude']
        self.num_parameters = self.SimulatedDataset.num_parameters + 2

        return

 
    def train(self, latitude=0, longitude=0):
        """
        Train the emulator for a particular point on the map.

        Parameters
        ----------
        latitude : float
            Latitude coordinate from the set of pixels we have in the
            SimulatedData attribute.
        longitude : float
            Corresponding longitude coordinate.

        Returns
        -------
        sklearn.gaussian_process.GaussianProcessRegressor
            A trained emulator for the specified pixel (if
            self.pixelwise==True) or the full region (if else).

        """
        true_data = self.true_data

        # Subset training data
        if self.pixelwise:

            # Select the data at the chosen pixel
            true_data = true_data[
                (true_data.latitude==latitude) & (true_data.longitude==longitude)
            ]

        # Choose columns for training input
        self.x_train = true_data.drop(
            labels=['response'] + list(
                self.exclude_features
            ),
                axis=1
            ).values

        # Choose column for training output. Record the mean for the purpose
        # of centering the response and not inflating the length scales.
        self.y_train = np.reshape(
            np.array(true_data.response),
            (-1,1)
        )
        self.y_mean = np.mean(self.y_train)

        # Declare the covariance function.
        kernel = 1.0 * Matern(
            length_scale=[1.0]*self.x_train.shape[1],
            length_scale_bounds=(1e-10, 1e20),
            nu=self.nu
        )

        # Fit the emulator. Center the response variable so as not to inflate
        # the length scales.
        self.emulator = GaussianProcessRegressor(kernel=kernel)
        self.emulator.fit(
            self.x_train,
            self.y_train - self.y_mean
        )

        return self.emulator


    def plot_pixel(
        self,
        fig_loc,
        latitude=0,
        longitude=0,
        parameter_of_interest=0,
        parameter_resolution=100
    ):
        """
        Visualize the emulator for a pixel.

        Parameters
        ----------
        latitude : float
            Latitude coordinate from the set of pixels we have in the
            SimulatedData attribute.
        longitude : float
            Corresponding longitude coordinate.
        parameter_resolution : integer, optional
            The resolution at which to draw the emulated response curve. The
            default is 100.

        Returns
        -------
        None.

        """
        num_parameters = self.num_parameters
        if self.pixelwise:
            num_parameters -= 2

        # Create test model variants. Sort the variants by the parameter of
        # interest (for when plotting the marginal response curve).
        x_test = pd.DataFrame(
            np.reshape(
                np.random.uniform(0, 1, num_parameters*parameter_resolution),
                (-1, num_parameters)
            )
        )
        x_test = x_test.sort_values(parameter_of_interest)

        # Get predictions.
        predictions, std = self.train(
            latitude=latitude,
            longitude=longitude
        ).predict(
            x_test,
            return_std=True
        )
        predictions = predictions.reshape((-1,)) + self.y_mean

        # Create a plot of the marginal response curve.
        plt.scatter(
            pd.DataFrame(self.x_train)[parameter_of_interest],
            pd.DataFrame(self.y_train),
            label='simulated data',
            c='orange'
        )
        plt.plot(
            x_test[parameter_of_interest],
            predictions,
            label='emulated curve',
            c='blue'
        )
        plt.fill_between(
            x_test[parameter_of_interest].ravel(),
            predictions - 1.96*std,
            predictions + 1.96*std,
            alpha = 0.5
        )

        # Make labels for the plot.
        plt.xlabel('parameter level')
        plt.ylabel('response')
        plt.title('Marginal response curve for parameter ' +
                  str(parameter_of_interest+1))
        plt.legend()
        plt.savefig(fig_loc)
        plt.close()
        return


    def emulate_variant(
            self,
            model_variant=[0]
        ):
        """
        Acquire the emulated response for a particular model variant over the
        whole map.

        Parameters
        ----------
        parameter_level : float, optional
            The level of the parameter at which a model variant is desired to
            be emulated. The default is 0.

        Returns
        -------
        pandas.DataFrame
            A data frame including the emulated response and emulator
            uncertainty (in standard deviation) over the map.

        """
        if (len(model_variant) != self.num_parameters - len(list(set(self.exclude_features)))):
            raise ValueError("The query variant does not match the " +
                             "dimension of the input space.")
            return

        # Build the dataframe that will hold the model variant data and
        # emulated response.
        self.model_variant = model_variant
        num_parameters = len(model_variant)
        
        pixels = self.SimulatedDataset.pixels
        variant_data = pixels.copy()
        variant_data[[
            'parameter' + str(digit) for digit in range(num_parameters)
        ]] = model_variant

        # Emulate across all pixels.
        if self.pixelwise:

            # First record predictions.
            variant_data[0] = [
                self.train(
                    latitude=row[0],
                    longitude=row[1]
                ).predict(
                    np.array(model_variant).reshape(1,-1),
                    return_std=True
                )
                    for row in pixels.to_numpy().tolist()
            ]

            # Separate the emulated response from the emulated standard
            # deviation. Note that the pixelwise emulators cannot estimate the
            # covariance between pixels.
            variant_data[['response', 'emulator_std']] = pd.DataFrame(
                variant_data[0].tolist()
            )
            variant_data['response'] = variant_data['response'].apply(
                lambda x: x[0]
            ) + self.y_mean
            variant_data['emulator_std'] = variant_data['emulator_std'].apply(
                lambda x: x[0]
            )

            return variant_data.drop(0, axis=1)

        # Emulate all pixels together.
        else:

            # First record predictions.
            response, cov = self.emulator.predict(
                variant_data.values,
                return_cov=True
            )

            # Add the emulated response and the emulated standard deviation to
            # the model variant dataframe.
            variant_data['response'] = pd.DataFrame(response) + self.y_mean
            variant_data = pd.concat(
                [variant_data, pd.DataFrame(cov)],
                axis=1
            )

            return variant_data


    def plot_variant(
            self,
            fig_loc,
            model_variant=[0],
            title_tag=None
        ):
        """
        Visualize a model variant.

        Parameters
        ----------
        parameter_level : float, optional
            The level of the parameter at which a model variant is desired to
            be emulated. The default is 0.

        Returns
        -------
        None.

        """
        
        self.model_variant = model_variant

        variant_data = self.emulate_variant(model_variant=model_variant)
        
        sc = plt.scatter(
            variant_data.longitude,
            variant_data.latitude,
            c=variant_data.response
        )
        plt.colorbar(sc)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        
        if title_tag is not None:
            plt.title(
                'Emulation at parameter level ' +
                str(title_tag)
            )
        plt.savefig(fig_loc)
        plt.close()
        return


def get_implausibility_from_least_squares_variant(obsSdCensor=np.inf):
    """
    Value

    Tuple : Variant which achieves least squares between measured and emulated AOD, "Distances" (differences in response)
        and "variances" (terms needed to normalize the distances)
    """
    which_gets_least_squares = []
    distances = []
    variances = []

    my_obs_df = obs_df.copy()
    my_obs_df.loc[obs_df.sdResponse >= obsSdCensor, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]

    # Get a best-variant for each day + time of day
    for time, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):

        my_obs_df_this_time = my_obs_df[my_obs_df.time==time].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index)

        with open(prediction_set, "r") as f:
            my_predict_df_this_time = pd.read_csv(
                f, index_col=0
            ).sort_values(
                ['time', 'longitude', 'latitude', 'variant']
            ).reset_index(
                drop=True
            )

        my_predict_dfs = [
            my_predict_df_this_time.iloc[k*5000:(k+1)*5000, :].reset_index(drop=True) 
            for k in range(num_pixels)
        ]

        # Check which row (test variant) gives least squares
        for row in range(num_pixels):

            y = my_obs_df_this_time.loc[row, 'meanResponse']
            e = my_obs_df_this_time.loc[row, 'sdResponse']**2

            zs = my_predict_dfs[row]['mean']
            ss = my_predict_dfs[row]['std']**2

            if ~np.isnan(y) and ~np.isnan(e) and y != 0 and e != 0:
                squares = list((y - zs)**2 / (e + ss))
                least_squares = min(squares)
                idx = squares.index(least_squares)

                which_gets_least_squares.append(idx)
                distances.append(y-zs[idx])
                variances.append(e + ss[idx])
            else:
                which_gets_least_squares.append(0)
                distances.append(float("nan"))
                variances.append(float("nan"))

    return (which_gets_least_squares, distances, variances)
