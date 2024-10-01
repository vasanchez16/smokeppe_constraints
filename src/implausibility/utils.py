import numpy as np
import netCDF4 as nc

def calculate_implausibility(nc_file_path, stats_dist_method, variance_mle, nu, epsilon):
    """
    doc
    """

    # get data
    open_nc_file = nc.Dataset(nc_file_path, 'r')
    my_distances = open_nc_file['distances'][:,:,:,:].data
    my_variances = open_nc_file['variances'][:,:,:,:].data
    variants = open_nc_file['variant'][:].data

    if epsilon != 0:
        my_distances = my_distances + epsilon

    my_variances_adjusted = my_variances + variance_mle

    if stats_dist_method == 'student-t':
        my_variances_adjusted = my_variances_adjusted * ((nu-2)/nu)

    implaus_arr = []
    for i in range(len(variants)):
        dists_variant_here = my_distances[:,:,:,i].flatten()
        varis_variant_here = my_variances_adjusted[:,:,:,i].flatten()

        implaus = np.sqrt(np.nansum((dists_variant_here ** 2) / varis_variant_here))
        implaus_arr.append([int(variants[i]),implaus])

    return implaus_arr


def calculate_implausibility2(variant, open_nc_file, stats_dist_method, **kwargs):
    """
    doc
    """
    

    variance_mle = kwargs['variance_mle']
    epsilon = kwargs['epsilon']

    # get data
    my_distances = open_nc_file['distances'][:,:,:,variant].data
    my_variances = open_nc_file['variances'][:,:,:,variant].data

    my_variances_adjusted = my_variances + variance_mle

    if stats_dist_method == 'student-t':
        nu_opt = kwargs['nu']
        my_variances_adjusted = my_variances_adjusted * ((nu_opt-2)/nu_opt)

    if epsilon != 0:
        my_distances = my_distances + epsilon

    implaus = np.sqrt(np.nansum((my_distances ** 2) / my_variances_adjusted, axis = (0,1,2)))

    implaus_arr = [variant, implaus]

    return implaus_arr
