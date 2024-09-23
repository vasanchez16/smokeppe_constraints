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
