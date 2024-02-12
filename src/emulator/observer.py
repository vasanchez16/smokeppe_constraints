# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:30:27 2022

@author: james

Instances of this class represent the measurements obtained from an instrument
supposed to have Gaussian noise.
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


class Observer:

    def __init__(
            self,
            SimulatedDataset,
            noise_level=0.1,
            omega=3.8,
            delta=12,
            emu_noise=True
        ):
        """
        Collect the parameters and data we need to define the measurements.

        Parameters
        ----------
        SimulatedDataset : SimulatedDataset
            The dataset that we need to emulate the simulator.
        noise_level : float, optional
            The standard deviation specified for the measurement error
            distribution. The default is 0.1.

        Returns
        -------
        None.

        """
        
        self.SimulatedDataset = SimulatedDataset
        self.noise_level = noise_level
        self.emu_noise = emu_noise
        self.omega = omega
        self.delta = delta


    def make_observations(
            self
        ):
        """
        Produce an instance of random observed data based on the truth of the
        SimulatedDataset.

        Returns
        -------
        None.

        """

        self.noise = np.random.normal(
            loc=0,
            scale=self.noise_level,
            size=self.SimulatedDataset.true_data.shape[0]
        ) + scipy.stats.t(
            df=self.omega,
            loc=0,
            scale=self.delta
        ).rvs(
            size=self.SimulatedDataset.true_data.shape[0]
        )
        self.SimulatedDataset.true_data[
            'observations'
        ] = self.SimulatedDataset.true_data.response + self.noise
        self.is_observed = True


    def plot_observations(self, fig_loc):
        """
        Visualize the measurements.

        Returns
        -------
        int
            Zero if observations have been made.

        """
        
        if not self.is_observed:
            return 0

        sc = plt.scatter(
            self.SimulatedDataset.true_data.longitude,
            self.SimulatedDataset.true_data.latitude,
            c=self.SimulatedDataset.true_data.observations
        )
        plt.colorbar(sc)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Observation of truth at noise level ' + str(self.noise_level))
        plt.savefig(fig_loc)
        plt.close()
        return


    def compute_statistics(self,
                           Emulator,
                           model_variants,
                           T):
        """
        Obtain statistics which can be used to accept or reject a model
        variant into a confidence set on the solution to the inverse problem.

        Parameters
        ----------
        Emulator : Emulator
            The emulator of study. The more confident is the emulator, the
            smaller is the denominator Var(y - z) and the less likely we are
            to accept a false model variant.
        model_variants : list of lists floats
            Levels of the parameter to test.
        T : function
            Method for computing statistic.

        Returns
        -------
        list of floats
            Statistics computed with T at each of the specified levels.

        """
        
        def y_hat(model_variant, pixelwise):
            """
            Obtain the emulated response at a specified parameter level.
    
            Parameters
            ----------
            parameter_level : float
                Value of calibration parameter.
    
            Returns
            -------
            list of floats
                Emulated response.
            list of floats
                Emulation standard deviation.

            """
            emulated = Emulator.emulate_variant(
                model_variant=model_variant
            )
            
            if pixelwise:
                return (emulated['response'],
                        emulated['emulator_std'])
            else:
                return (emulated['response'],
                        emulated.drop(
                            labels=['latitude', 'longitude', 'response'] +
                            list(emulated.filter(regex='parameter')),
                                      axis=1))

        pixelwise = Emulator.pixelwise
        z = self.SimulatedDataset.true_data.observations

        e = np.array([self.noise_level] * len(z))

        return [
            T(*y_hat(variant, pixelwise),
              z,
              e,
              emu_noise=self.emu_noise) for variant in model_variants
        ]


    def plot_statistics(self, Emulator, model_variants, param_of_interest, T, fig_loc):
        """
        Show how the strict bound statistics for our example compare to a high
        quantile from the chi-square distribution -- this is the distribution
        which the statistic computed at the true parameter level should
        follow.

        Parameters
        ----------
        Emulator : Emulator
            The emulator of study. The more confident is the emulator, the
            smaller is the denominator Var(y - z) and the less likely we are
            to accept a false model variant.
        model_variants : list of lists floats
            Levels of the parameter to test.
        param_of_interest: int
            The index for the entry in a model variant the plot of whose
            plausible region is desired.
        T : function
            Method for computing statistic.

        Returns
        -------
        None.

        """

        pixelwise = Emulator.pixelwise

        # Compute the statistics
        my_statistics = self.compute_statistics(
            Emulator,
            model_variants,
            T
        )
        critical_value = scipy.stats.chi2.ppf(
            0.95,
            self.SimulatedDataset.pixels.shape[0]
        )

        # Distinguish between the accepted and rejected pixels
        plot_data = pd.DataFrame({'param' : model_variants,
                                  'stat' : my_statistics})
        plot_data_accept = plot_data[plot_data.stat <= critical_value].copy()
        plot_data_reject = plot_data[plot_data.stat > critical_value].copy()

        # Create plot
        plt.scatter([variant[param_of_interest]
                     for variant in plot_data_accept.param],
                    plot_data_accept.stat,
                    c='green',
                    label='Plausible')
        plt.scatter([variant[param_of_interest]
                     for variant in plot_data_reject.param],
                    plot_data_reject.stat,
                    c='red',
                    label='Implausible')

        # Add labels
        plt.legend()
        plt.axhline(y=critical_value)
        plt.title('Strict bound statistics compared to a chi-square '+
                  'quantile \n Emulation pixelwise: ' + str(pixelwise) +
                  '\n Include emulation uncertainty ' + str(self.emu_noise))
        plt.savefig(fig_loc)
        plt.close()
        return


    def compare_statistics_plot(
            self,
            Emulator,
            model_variants,
            param_of_interest,
            fig_loc,
            plot_lines=True
        ):
        
        if Emulator.pixelwise:
            raise ValueError("Supply a total emulator, not pixelwise.")
            return
        elif not self.is_observed:
            raise ValueError("Observations unmade.")
            return

        import custom_statistics as stat
        import matplotlib.lines as mlines

        # Compute the statistics
        hm_statistics = self.compute_statistics(
            Emulator,
            model_variants,
            stat.history_matching_statistic
        )
        sb_statistics = self.compute_statistics(
            Emulator,
            model_variants,
            stat.strict_bounds_statistic_from_total_emulator
        )
        
        # Compute critical values
        sb_cv = scipy.stats.chi2.ppf(
            0.95,
            self.SimulatedDataset.pixels.shape[0]
        )
        hm_cv = self.noise_level*1.5*np.sqrt(
            2*np.log(self.SimulatedDataset.pixels.shape[0])
        )

        # Make table
        # Distinguish between the accepted and rejected pixels
        hm_data = pd.DataFrame({'param' : [variant[param_of_interest] for variant in model_variants],
                                  'stat' : hm_statistics,
                                  'accept' : False,
                                  'hm' : True})
        hm_data['accept'] = (hm_data.stat <= hm_cv)
        hm_data = hm_data.sort_values('param')
        
        sb_data = pd.DataFrame({'param' : [variant[param_of_interest] for variant in model_variants],
                                  'stat' : sb_statistics,
                                  'accept' : False,
                                  'hm' : False})
        sb_data['accept'] = (sb_data.stat <= sb_cv)
        sb_data = sb_data.sort_values('param')
        
        plot_data = pd.concat([hm_data, sb_data], axis=0).reset_index(drop=True)        
        
        accept = plot_data.accept
        hm = plot_data.hm

        # Create plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # fig.set_figheight(10)
        # fig.set_figwidth(20)
        
        ax1.scatter(plot_data.param[accept*(-hm)],
            plot_data.stat[accept*(-hm)],
            c='blue',
            marker='^')

        ax1.scatter(plot_data.param[(-accept)*(-hm)],
                    plot_data.stat[(-accept)*(-hm)],
                    c='blue',
                    marker='o')
        
        ax1.axhline(y=sb_cv, c='b', ls='--')
        
        ax2.scatter(plot_data.param[accept*hm],
                    plot_data.stat[accept*hm],
                    c='red',
                    marker='^')
        
        ax2.scatter(plot_data.param[(-accept)*hm],
                    plot_data.stat[(-accept)*hm],
                    c='red',
                    marker='o')

        ax2.axhline(y=hm_cv, c='r', ls='--')

        if plot_lines:
            ax1.plot(plot_data.param[(-hm)],
                     plot_data.stat[(-hm)],
                     c='blue')
            ax2.plot(plot_data.param[(hm)],
                 plot_data.stat[(hm)],
                 c='red')

        # Add labels
        strict_bounds = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                  markersize=5, label='Strict bounds')
        history_matching = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                  markersize=5, label='History matching')
        implausible = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                  markersize=5, label='Implausible')
        plausible = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                  markersize=5, label='Plausible')
        
        fig.legend(
            handles=[strict_bounds, history_matching, implausible, plausible],
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=4
        )
        
        ax1.set_title('Implausibilities under different metrics')
        ax1.set_xlabel('parameter ' + str(param_of_interest) + ' level of model variant')
        ax1.set_ylabel('strict bound metric')
        ax2.set_ylabel('history matching metric')
        fig.savefig(fig_loc)
        plt.close()
        return
