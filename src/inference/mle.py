import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def get_implausibility_from_least_squares_variant(obsSdCensor=1):
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


my_obs_df = outliers_new.copy()

for time, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):

    my_obs_df_this_time = my_obs_df[my_obs_df.time==time].reset_index(drop=True)
    num_pixels = len(my_obs_df_this_time.index)

    with open(prediction_set, "r") as f:
        my_predict_df_this_time = pd.read_csv(f, index_col=0).sort_values(
            ['time', 'longitude', 'latitude', 'variant']
        ).reset_index(
            drop=True
        )

    my_predict_dfs = [
        my_predict_df_this_time.iloc[k*5000:(k+1)*5000, :].reset_index(drop=True) 
        for k in range(num_pixels)
    ]

    for row in range(num_pixels):
        print(my_predict_dfs[row].loc[:, ['latitude', 'longitude', 'time']])
        print(my_obs_df_this_time.loc[row, ['latitude', 'longitude', 'time']])



def get_all_squares(outlier_set=outliers_new, method='sb'):
    """
    y = observed AOD
    zs = emulated AODs (vector)

    e = estimated instrument error standard deviation
    ss = standard deviation of AOD emulation


    Arguments

    idxSet : Set of row indices which are to be excluded from analysis
    method : Which method to use for estimating uncertainties, either 'sb' (strict bounds, our method) or 'hm' (history
        matching, based on Johnson et al. (2020))


    Value
    
    Tuple : "Distances" (differences in response) and "variances" (terms needed to normalize the distances)
    """
    allDistances = []
    allVariances = []

    idxSet=list((outlier_set['missing']) | (outlier_set['outlier']))
    my_obs_df = outlier_set.copy()
    my_obs_df.loc[idxSet, ["meanResponse", "sdResponse"]] = [float("nan"), float("nan")]

    for time, prediction_set in zip(np.unique(my_obs_df.time), prediction_sets):
        
        my_obs_df_this_time = my_obs_df[my_obs_df.time==time].reset_index(drop=True)
        num_pixels = len(my_obs_df_this_time.index)
        
        with open(prediction_set, "r") as f:
            my_predict_df_this_time = pd.read_csv(f, index_col=0).sort_values(
                ['time', 'longitude', 'latitude', 'variant']
            ).reset_index(
                drop=True
            )
        
        my_predict_dfs = [
            my_predict_df_this_time.iloc[k*5000:(k+1)*5000, :].reset_index(drop=True) 
            for k in range(num_pixels)
        ]

        for row in range(num_pixels):

            y = my_obs_df_this_time.loc[row, 'meanResponse']
            if method=='sb':
                e = my_obs_df_this_time.loc[row, 'sdResponse']**2
            elif method=='hm':
                # Per Johnson et al. (2020), instrument uncertainty is 10%, spatial co-location uncertainty is 20%, and
                # temporal sampling uncertainty is 10% of the measured value.
                e = (0.1+0.2+0.1)*y

            zs = my_predict_dfs[row]['mean']
            ss = my_predict_dfs[row]['std']**2

            if ~np.isnan(y) and y != 0:
                distances = list(y - zs)
                variances = list(e + ss)
            else:
                distances = [float('nan')]*len(zs)
                variances = [float('nan')]*len(zs)

            allDistances.append(pd.DataFrame(distances).transpose())
            allVariances.append(pd.DataFrame(variances).transpose())

    return (
        pd.concat(allDistances, axis=0).reset_index(drop=True),
        pd.concat(allVariances, axis=0).reset_index(drop=True)
    )


def l_new(d, u):
    # negative Log likelihood, to be minimized
    term1 = np.nansum(np.log(my_variances_new.iloc[:, u] + d**2))
    term2 = np.nansum(np.power(my_distances_new.iloc[:, u], 2) / (my_variances_new.iloc[:, u] + d**2))
    return 0.5 * (term1 + term2)


max_l_for_us_new = []

for u in range(5000):
    res = minimize_scalar(lambda d: l_new(d, u))
    max_l_for_us_new.append(-res.fun)


with open('mle_new', 'w') as f:
    write = csv.writer(f)
    write.writerow([u_mle_new, additional_variance_new])
f.close()