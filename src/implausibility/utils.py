import numpy as np
import netCDF4 as nc


def calculate_implausibility(nc_file_path, stats_dist_method, variance_mle, nu, epsilon):
    """
    Compute the implausibility statistic for all variants in a single data file.

    For each variant k, the implausibility is:

        I(u^k) = sqrt( sum_i  (d_i - epsilon)^2 / v_i_adj )

    where d_i are distances, v_i_adj = variances + variance_mle (scaled by
    (nu-2)/nu for the student-t method), and the sum runs over all non-NaN
    spatial/temporal points.

    Arguments:
    - nc_file_path: str, path to a distances_variances_*.nc file
    - stats_dist_method: str, one of the supported method names (e.g. 'student-t')
    - variance_mle: float, model discrepancy variance estimated by MLE
    - nu: float, degrees of freedom (used only for 'student-t' method)
    - epsilon: float, mean bias term (0 if not estimated)

    Returns:
    - list of [variant_str, implausibility_str] pairs for all variants in the file
    """
    file_str = nc_file_path.split('/')[-1]
    print(f'Starting {file_str}...')

    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        my_distances = open_nc_file['distances'][:, :, :, :].data
        my_variances = open_nc_file['variances'][:, :, :, :].data
        variants = open_nc_file['variant'][:].data

    if epsilon != 0:
        my_distances = my_distances - epsilon

    my_variances_adjusted = my_variances + variance_mle

    if stats_dist_method == 'student-t':
        my_variances_adjusted = my_variances_adjusted * ((nu - 2) / nu)

    implaus = np.sqrt(np.nansum((my_distances ** 2) / my_variances_adjusted, axis=(0, 1, 2)))

    print(f'{file_str} complete.')
    return [[str(int(variant)), str(i)] for variant, i in zip(variants, implaus)]
