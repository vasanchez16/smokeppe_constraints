import pandas as pd
import numpy as np
import os
import json
import netCDF4 as nc
from datetime import datetime


def save_dataset(data, save_path):
    """
    Save a pandas DataFrame to CSV.

    Arguments:
    - data: pd.DataFrame to save
    - save_path: str, destination file path
    """
    data.to_csv(save_path, index=False)


def get_variant_subsets(num_variants, subset_size):
    """
    Partition variant indices into fixed-size chunks.

    Returns a list of lists, each of length subset_size (the last chunk may be smaller).
    """
    variants_list = list(range(num_variants))
    number_of_subsets = int(np.ceil(num_variants / subset_size))
    return [variants_list[i * subset_size:(i + 1) * subset_size] for i in range(number_of_subsets)]


def create_distances_and_variances_base_files(save_here_dir, obs_df, variant_subset):
    """
    Create an empty NetCDF file to hold distances and variances for a variant subset.

    Dimensions: (time unlimited, lat, lon, variant).
    Latitude, longitude, and variant coordinate variables are populated immediately;
    the time, distances, and variances variables are filled incrementally by
    save_distances_and_variances_one_time.

    Arguments:
    - save_here_dir: str, run output directory
    - obs_df: pd.DataFrame, used to extract unique lat/lon coordinates
    - variant_subset: list of int, variant indices this file will store
    """
    max_variant = max(variant_subset)
    file_path = save_here_dir + 'dists_varis_data/' + f'distances_variances_{max_variant}.nc'

    with nc.Dataset(file_path, mode='w', format='NETCDF4') as nc_file:
        nc_file.createDimension('lat', len(np.unique(obs_df['latitude'])))
        nc_file.createDimension('lon', len(np.unique(obs_df['longitude'])))
        nc_file.createDimension('variant', len(variant_subset))
        nc_file.createDimension('time', None)  # unlimited

        lats = nc_file.createVariable('latitude', 'f4', ('lat',))
        lons = nc_file.createVariable('longitude', 'f4', ('lon',))
        variants = nc_file.createVariable('variant', 'i4', ('variant',))
        times = nc_file.createVariable('time', 'f4', ('time',))
        dists = nc_file.createVariable('distances', 'f4', ('time', 'lat', 'lon', 'variant'))
        varis = nc_file.createVariable('variances', 'f4', ('time', 'lat', 'lon', 'variant'))

        lats.units = 'degrees north'
        lons.units = 'degrees east'
        dists.units = 'Observation - Emulator'
        varis.units = 'Observation variance + Emulator variance'
        times.units = 'Hours since 01-01-1900 T00:00:00'

        lats[:] = np.unique(obs_df['latitude'])
        lons[:] = np.unique(obs_df['longitude'])
        variants[:] = variant_subset


def save_distances_and_variances_one_time(save_here_dir, dists_one_time, varis_one_time, obs_time, index, variant_subsets):
    """
    Append distances and variances for a single time step to each variant-subset NetCDF file.

    Arguments:
    - save_here_dir: str, run output directory
    - dists_one_time: array-like of shape (lat, lon, all_variants), distances for this time step
    - varis_one_time: array-like of shape (lat, lon, all_variants), variances for this time step
    - obs_time: str, raw timestamp string (e.g. '2005-01-01 00:00:00')
    - index: int, time axis index to write to
    - variant_subsets: list of lists, variant index groupings matching the on-disk files
    """
    dists_one_time = np.array(dists_one_time)
    varis_one_time = np.array(varis_one_time)
    adj_time = get_adj_time(obs_time)

    for subset in variant_subsets:
        max_variant = max(subset)
        file_path = save_here_dir + 'dists_varis_data/' + f'distances_variances_{max_variant}.nc'

        with nc.Dataset(file_path, mode='a') as nc_file:
            nc_file.variables['time'][index:index + 1] = np.array([adj_time])
            nc_file.variables['distances'][index, :, :, :] = dists_one_time[:, :, subset]
            nc_file.variables['variances'][index, :, :, :] = varis_one_time[:, :, subset]


def save_distances_and_variances(save_here_dir, distances, variances, obs_df, num_variants):
    """
    Save the full distances and variances arrays to a single NetCDF file.

    Arguments:
    - save_here_dir: str, run output directory
    - distances: np.ndarray of shape (time, lat, lon, variant)
    - variances: np.ndarray of shape (time, lat, lon, variant)
    - obs_df: pd.DataFrame, used to extract coordinate and time arrays
    - num_variants: int, number of emulator parameter variants
    """
    nc_file = nc.Dataset(save_here_dir + 'distances_variances.nc', 'w', format='NETCDF4')

    nc_file.createDimension('lat', len(obs_df['latitude'].unique()))
    nc_file.createDimension('lon', len(obs_df['longitude'].unique()))
    nc_file.createDimension('variant', num_variants)
    nc_file.createDimension('time', len(np.unique(obs_df['time'])))

    lats = nc_file.createVariable('latitude', 'f4', ('lat',))
    lons = nc_file.createVariable('longitude', 'f4', ('lon',))
    variants = nc_file.createVariable('variant', 'i4', ('variant',))
    times = nc_file.createVariable('time', 'f4', ('time',))
    dists = nc_file.createVariable('distances', 'f4', ('time', 'lat', 'lon', 'variant'))
    varis = nc_file.createVariable('variances', 'f4', ('time', 'lat', 'lon', 'variant'))

    lats.units = 'degrees north'
    lons.units = 'degrees east'
    dists.units = 'Observation - Emulator'
    varis.units = 'Observation variance + Emulator variance'
    times.units = 'Hours since 01-01-1900 T00:00:00'

    lats[:] = obs_df['latitude'].unique()
    lons[:] = obs_df['longitude'].unique()
    variants[:] = list(range(num_variants))
    times[:] = get_times_for_nc(np.unique(obs_df['time']))

    dists[:, :, :, :] = distances
    varis[:, :, :, :] = variances
    nc_file.description = 'Distances and Variances values for constraint calculations.'
    nc_file.close()


def get_adj_time(raw_date):
    """Convert a datetime string to hours elapsed since 1900-01-01 00:00:00."""
    basetime = datetime(1900, 1, 1, 0, 0, 0)
    adj_time = datetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S') - basetime
    return adj_time.total_seconds() / 3600


def get_times_for_nc(raw_times):
    """Convert an array of datetime strings to hours since 1900-01-01 00:00:00."""
    basetime = datetime(1900, 1, 1, 0, 0, 0)
    return [
        (datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - basetime).total_seconds() / 3600
        for t in raw_times
    ]


def save_indexed_dataset():
    """Save distances for a specific parameter set. Not yet implemented."""
    raise NotImplementedError


def set_up_directories(args):
    """
    Create the output directory structure for a run.

    Creates the run-label directory and all required subdirectories
    (implaus_figures, general_figures, general_figures/movie_pngs, dists_varis_data)
    if they do not already exist.
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    base = args.output_dir + run_label

    for path in [
        base,
        base + '/implaus_figures',
        base + '/general_figures',
        base + '/general_figures/movie_pngs',
        base + '/dists_varis_data',
    ]:
        if not os.path.exists(path):
            os.mkdir(path)


def set_up_directories_combined_implaus(args):
    """
    Create the output directory structure for a combined-implausibility run.

    In addition to the base and implaus_figures directories, creates a
    comb_implaus_figures directory and per-run subdirectories under implaus_figures.
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']
    run_dirs = eval_params['directories']
    base = args.output_dir + run_label

    for path in [
        base,
        base + '/comb_implaus_figures',
        base + '/implaus_figures',
    ]:
        if not os.path.exists(path):
            os.mkdir(path)

    for dir in run_dirs:
        subfolder = dir.split('/')[-1]
        subpath = base + '/implaus_figures/' + subfolder
        if not os.path.exists(subpath):
            os.mkdir(subpath)


def save_eval_params_file(args):
    """Copy the evaluation parameters JSON into the run output directory."""
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    with open(args.output_dir + run_label + '/evaluationParameters.json', 'w') as json_file:
        json.dump(eval_params, json_file, indent=4)


def run_checks(args):
    """
    Validate required evaluation parameters before the pipeline starts.

    Raises ValueError for:
    - An unrecognised stats_distribution_method
    - output_dir or emulator_output_folder_path not ending with '/'
    - confidence_level outside the range (0, 100)
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    possible_methods = [
        'convolution',
        'student-t',
        'gaussian',
        'student-t_bootstrap',
        'gaussian_bootstrap',
    ]
    if eval_params['stats_distribution_method'] not in possible_methods:
        raise ValueError("Method must be one of: 'convolution', 'student-t', 'gaussian', "
                         "'student-t_bootstrap', 'gaussian_bootstrap'")

    if args.output_dir[-1] != '/':
        raise ValueError("End output_dir with '/' character")

    if eval_params['emulator_output_folder_path'][-1] != '/':
        raise ValueError("End emulator_output_folder_path with '/' character")

    try:
        conf_lvl = float(eval_params['confidence_level'])
        if conf_lvl >= 100 or conf_lvl <= 0:
            raise ValueError('confidence_level must be a number between 0 and 100')
    except KeyError:
        pass  # confidence_level is optional; a default is applied downstream


def runtime(seconds):
    """Format an elapsed time in seconds as 'X hours Y minutes'."""
    hrs = int(seconds / (60 * 60))
    minutes = int((seconds % (60 * 60)) / 60)
    return f'Current Runtime: {hrs} hours {minutes} minutes'


def get_em_pred_filenames(args):
    """
    Return a sorted list of emulator prediction filenames (.nc or .csv) from the
    emulator output folder specified in the evaluation parameters file.
    """
    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)
    emulator_folder_path = eval_params['emulator_output_folder_path']

    folder_contents = os.listdir(emulator_folder_path)
    folder_contents = [f for f in folder_contents if f.endswith('.nc') or f.endswith('.csv')]
    folder_contents.sort()
    return folder_contents


def get_mle_columns(init_vals):
    """
    Return the MLE result column names based on the number of optimized parameters.

    Two initial values (sigma, nu) → no epsilon column.
    Three or more (sigma, nu, epsilon) → epsilon column included.
    """
    if len(init_vals) > 2:
        return ['parameter_set_num', 'variance_mle', 'nu', 'epsilon', 'log_L']
    return ['parameter_set_num', 'variance_mle', 'nu', 'log_L']


def create_mle_base_file(all_mle_file, cols_here):
    """Create an empty NetCDF file with an unlimited parameter_set_num dimension and one variable per column."""
    with nc.Dataset(all_mle_file, mode='w', format='NETCDF4') as nc_file:
        nc_file.createDimension('parameter_set_num', None)
        for col in cols_here:
            nc_file.createVariable(col, 'f4', ('parameter_set_num',))


def save_mle_to_nc(save_here_dir, CSIZE, cols_here):
    """
    Merge per-rank MLE CSV files into a single all_mle.nc NetCDF file, then
    delete the per-rank CSVs.

    Each rank file is sorted by parameter_set_num before being appended so that
    the final NetCDF has variants in a consistent order.
    """
    all_mle_file = os.path.join(save_here_dir, 'all_mle.nc')
    create_mle_base_file(all_mle_file, cols_here)

    offset = 0
    for rank_num in range(CSIZE):
        rank_file = os.path.join(save_here_dir, f'mle_res_rank{rank_num}.csv')
        mle_df = pd.read_csv(rank_file)
        mle_df.sort_values(by='parameter_set_num', inplace=True, ignore_index=True)
        n_rows = len(mle_df)

        with nc.Dataset(all_mle_file, 'a') as nc_file:
            for col in cols_here:
                nc_file.variables[col][offset:offset + n_rows] = mle_df[col].values
            nc_file.sync()

        try:
            os.remove(rank_file)
        except Exception as e:
            print(f'Warning: could not delete {rank_file}: {e}', flush=True)

        offset += n_rows
