import numpy as np

def calculate_implausibility(variant, open_nc_file, stats_dist_method, **kwargs):
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
