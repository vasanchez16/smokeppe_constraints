import numpy as np
import netCDF4 as nc

def calculate_implausibility(nc_file_path, stats_dist_method, variance_mle, nu, epsilon):
    """
    doc
    """
    file_str = nc_file_path.split('/')[-1]

    print(f'Starting {file_str}...')
    # get data
    with nc.Dataset(nc_file_path, 'r') as open_nc_file:
        my_distances = open_nc_file['distances'][:,:,:,:].data
        my_variances = open_nc_file['variances'][:,:,:,:].data
        variants = open_nc_file['variant'][:].data

    if epsilon != 0:
        my_distances = my_distances + epsilon

    my_variances_adjusted = my_variances + variance_mle

    if stats_dist_method == 'student-t':
        my_variances_adjusted = my_variances_adjusted * ((nu-2)/nu)

    implaus = np.sqrt(np.nansum((my_distances ** 2) / my_variances_adjusted, axis = (0,1,2)))

    implaus_arr = [[str(int(variant)), str(i)] for variant, i in zip(variants,implaus)]
    
    print(f'{file_str} complete.')

    return implaus_arr