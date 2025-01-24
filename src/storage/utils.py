import pandas as pd
import numpy as np
import os
import json
import netCDF4 as nc
from datetime import datetime


def save_dataset(data, save_path):
    """
    Arguments:
    data: pandas DataFrame Obj
    Data to be saved
    save_path: str
    Path where this data will be saved
    """
    data.to_csv(save_path, index=False)
    return

def get_variant_subsets(num_variants, subset_size):
    variants_list = list(range(num_variants))

    number_of_subsets = int(np.ceil(num_variants / subset_size))

    variant_subsets = []
    for i in range(number_of_subsets):
        variant_subsets.append(variants_list[i*subset_size:(i+1)*subset_size])
    
    return variant_subsets

def create_distances_and_variances_base_files(save_here_dir, obs_df, variant_subset):
    """
    Doc
    """
    max_variant = max(variant_subset)
    with nc.Dataset(save_here_dir + 'dists_varis_data/' + f'distances_variances_{max_variant}.nc', mode="w", format="NETCDF4") as nc_file:
        # Create dimensions
        nc_file.createDimension("lat", len(np.unique(obs_df['latitude'])))
        nc_file.createDimension("lon", len(np.unique(obs_df['longitude'])))
        nc_file.createDimension("variant", len(variant_subset))
        nc_file.createDimension("time", None)  # None for unlimited time dimension

        # Create variables
        lats = nc_file.createVariable('latitude', 'f4', ('lat',))
        lons = nc_file.createVariable('longitude', 'f4', ('lon',))
        variants = nc_file.createVariable('variant', 'i4', ('variant',))
        times = nc_file.createVariable('time', 'f4', ('time',))
        dists = nc_file.createVariable('distances', 'f4', ('time', 'lat', 'lon', 'variant'))
        varis = nc_file.createVariable('variances', 'f4', ('time', 'lat', 'lon', 'variant'))

        # Define units for variables
        lats.units = 'degrees north'
        lons.units = 'degrees east'
        dists.units = 'Observation - Emulator' 
        varis.units = 'Observation variance + Emulator variance'
        times.units = 'Hours since 01-01-1900 T00:00:00'
        
        # Initialize the lat, lon, and variant arrays
        lats[:] = np.unique(obs_df['latitude'])  # Fill latitudes
        lons[:] = np.unique(obs_df['longitude']) # Fill longitudes
        variants[:] = variant_subset

    return None

def create_distances_and_variances_base_files_nongridded(save_here_dir, obs_df, variant_subset):
    """
    Doc
    """
    max_variant = max(variant_subset)
    with nc.Dataset(save_here_dir + 'dists_varis_data/' + f'distances_variances_{max_variant}.nc', mode="w", format="NETCDF4") as nc_file:
        # Create dimensions
        nc_file.createDimension("variant", len(variant_subset))
        nc_file.createDimension("time", None)  # None for unlimited time dimension

        # Create variables
        variants = nc_file.createVariable('variant', 'i4', ('variant',))
        times = nc_file.createVariable('time', 'i4', ('time',))
        dists = nc_file.createVariable('distances', 'f4', ('time', 'variant'))
        varis = nc_file.createVariable('variances', 'f4', ('time', 'variant'))

        # Define units for variables
        dists.units = 'Observation - Emulator' 
        varis.units = 'Observation variance + Emulator variance'
        times.units = 'flight number'
        
        # Initialize the lat, lon, and variant arrays
        variants[:] = variant_subset

    return None

def save_distances_and_variances_one_time(save_here_dir, dists_one_time, varis_one_time, obs_time, index, variant_subsets):
    """
    doc
    """
    for subset in variant_subsets:
        max_variant = max(subset)

        with nc.Dataset(save_here_dir + 'dists_varis_data/' +  f'distances_variances_{max_variant}.nc', mode="a") as nc_file:
            # Append the time value
            adj_time = get_adj_time(obs_time)
            nc_file.variables["time"][index:index+1] = np.array([adj_time])
            
            dists_one_time = np.array(dists_one_time)
            varis_one_time = np.array(varis_one_time)

            # Append the data for this time step
            nc_file.variables["distances"][index, :, :, :] = dists_one_time[:,:,subset]
            nc_file.variables["variances"][index, :, :, :] = varis_one_time[:,:,subset]


    return None

def save_distances_and_variances_one_time_nongridded(save_here_dir, dists_one_time, varis_one_time, obs_time, flight_index, variant_subsets):
    """
    doc
    """
    for subset in variant_subsets:
        max_variant = max(subset)
        print(subset)

        with nc.Dataset(save_here_dir + 'dists_varis_data/' +  f'distances_variances_{max_variant}.nc', mode="a") as nc_file:
            dists_one_time = np.array(dists_one_time)
            varis_one_time = np.array(varis_one_time)
            # get list of times equal in length to number of points in dists_one_time
            num_points = dists_one_time.shape[0]
            flight_num = [flight_index + 1] * num_points
            # append flight num to existing data in time variable
            nc_file.variables["time"][flight_index*num_points:(flight_index*num_points)+num_points] = np.array(flight_num)

            # Append the data for this time step
            nc_file.variables["distances"][flight_index*num_points:(flight_index*num_points)+num_points, :] = dists_one_time[:,subset]
            nc_file.variables["variances"][flight_index*num_points:(flight_index*num_points)+num_points, :] = varis_one_time[:,subset]


    return None

def save_distances_and_variances(save_here_dir, distances, variances, obs_df, num_variants):
    """
    Saves the distances and variances calculations into netCDF files.
    """
    nc_file = nc.Dataset(save_here_dir + 'distances_variances.nc', 'w', format='NETCDF4')

    # Create dimensions
    nc_file.createDimension('lat', len(obs_df['latitude'].unique()))
    nc_file.createDimension('lon', len(obs_df['longitude'].unique()))
    nc_file.createDimension('variant', num_variants)
    nc_file.createDimension('time', len(np.unique(obs_df['time'])))


    # Create variables
    lats = nc_file.createVariable('latitude', 'f4', ('lat',))
    lons = nc_file.createVariable('longitude', 'f4', ('lon',))
    variants = nc_file.createVariable('variant', 'i4', ('variant',))
    times = nc_file.createVariable('time', 'f4', ('time',))
    dists = nc_file.createVariable('distances', 'f4', ('time', 'lat', 'lon', 'variant'))
    varis = nc_file.createVariable('variances', 'f4', ('time', 'lat', 'lon', 'variant'))

    # Define units for variables
    lats.units = 'degrees north'
    lons.units = 'degrees east'
    dists.units = 'Observation - Emulator' 
    varis.units = 'Observation variance + Emulator variance'
    times.units = 'Hours since 01-01-1900 T00:00:00'

    # Write data to the variables
    lats[:] = obs_df['latitude'].unique()  # Fill latitudes
    lons[:] = obs_df['longitude'].unique() # Fill longitudes
    variants[:] = list(range(num_variants))

    # conv times to format for nc file
    norm_times = get_times_for_nc(np.unique(obs_df['time']))
    times[:] = norm_times

    # Store dists and varis data
    dists[:,:,:,:] = distances
    varis[:,:,:,:] = variances
    # Add a global attribute
    nc_file.description = 'Distances and Variances values for constraint calculations.'
    nc_file.close()
    return None

def get_adj_time(raw_date):
    basetime = datetime(1900,1,1,0,0,0,0)

    adj_time = datetime.strptime(raw_date,'%Y-%m-%d %H:%M:%S') - basetime

    adj_time = adj_time.total_seconds() / 3600

    return adj_time

def get_times_for_nc(raw_times):
    basetime = datetime(1900,1,1,0,0,0,0)
    time_norm_func = lambda t: datetime.strptime(t,'%Y-%m-%d %H:%M:%S') - basetime

    norm_times = []
    for t in raw_times:
        time_diff = time_norm_func(t)
        norm_times.append(time_diff.total_seconds() / 3600)
    
    return norm_times

def save_indexed_dataset():
    """
    Save distances separately for specific parameter set.
    Implement later if needed
    """
    raise NotImplementedError

def set_up_directories(args):
    """
    add doc
    """
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    if not os.path.exists(args.output_dir + run_label):
        os.mkdir(args.output_dir + run_label)

    if not os.path.exists(args.output_dir + run_label + '/implaus_figures'):
        os.mkdir(args.output_dir + run_label + '/implaus_figures')

    if not os.path.exists(args.output_dir + run_label + '/general_figures'):
        os.mkdir(args.output_dir + run_label + '/general_figures')
    
    if not os.path.exists(args.output_dir + run_label + '/general_figures/movie_pngs'):
        os.mkdir(args.output_dir + run_label + '/general_figures/movie_pngs')

    if not os.path.exists(args.output_dir + run_label + '/dists_varis_data'):
        os.mkdir(args.output_dir + run_label +  '/dists_varis_data')

    return

def set_up_directories_combined_implaus(args):
    """
    add doc
    """
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    run_dirs = eval_params['directories']

    if not os.path.exists(args.output_dir + run_label):
        os.mkdir(args.output_dir + run_label)

    if not os.path.exists(args.output_dir + run_label + '/comb_implaus_figures'):
        os.mkdir(args.output_dir + run_label + '/comb_implaus_figures')

    if not os.path.exists(args.output_dir + run_label + '/implaus_figures'):
        os.mkdir(args.output_dir + run_label + '/implaus_figures')

    for dir in run_dirs:
        subfolder = dir.split('/')[-1]
        if not os.path.exists(args.output_dir + run_label + '/implaus_figures/' + subfolder):
            os.mkdir(args.output_dir + run_label + '/implaus_figures/' + subfolder)

    return

def save_eval_params_file(args):
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    with open(args.output_dir + run_label + '/evaluationParameters.json','w') as json_file:
        json.dump(eval_params, json_file, indent=4)

    return

def run_checks(args):
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    possible_methods = [
        'convolution',
        'student-t',
        'gaussian',
        'student-t_bootstrap',
        'gaussian_bootstrap'
                        ]
    if not (eval_params['stats_distribution_method'] in possible_methods):
        raise(ValueError('Method must be one of the following: \'convolution\',\'student-t\',\'gaussian\''))

    if (args.output_dir[-1] != '/') and (eval_params['emulator_output_folder_path'][-1] != '/'):
        raise(ValueError('End OutputDir and emulator_output_folder_path with \'/\' character'))

    if args.output_dir[-1] != '/':
        raise(ValueError('End OutputDir with \'/\' character'))
    
    if eval_params['emulator_output_folder_path'][-1] != '/':
        raise(ValueError('End emulator_output_folder_path with \'/\' character'))
    
    try:
        eval_params['confidence_level']
        if (float(eval_params['confidence_level']) >= 100) or (float(eval_params['confidence_level']) <= 0):
            raise(ValueError('Confidence level must be a number between 0 and 100'))
    except:
        None
    
    return

def runtime(seconds):
    hrs = int(seconds / (60*60))
    minutes = int((seconds % (60*60)) / 60)
    return f'Current Runtime: {hrs} hours {minutes} minutes'

def get_em_pred_filenames(args):
    """
    getting sorted list of the em prediciton filenames
    """

    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    emulator_folder_path = eval_params['emulator_output_folder_path']

    folder_contents = os.listdir(emulator_folder_path)
    folder_contents.sort()

    return folder_contents
