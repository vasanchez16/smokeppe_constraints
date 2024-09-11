import pandas as pd
import os
import json


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
    
    return

def runtime(seconds):
    hrs = int(seconds / (60*60))
    minutes = int((seconds % (60*60)) / 60)
    return f'Current Runtime: {hrs} hours {minutes} minutes'
