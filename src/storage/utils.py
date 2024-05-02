import pandas as pd
import os
import json


def set_up_directories(args):
    results_dir = args.output_dir
    fig_dir = results_dir + 'fig/'
    sim_loc = results_dir + 'sim_data.csv'
    obs_loc = results_dir + 'obs_data.csv'
    params_loc = results_dir + 'params_data.csv'
    emu_loc = results_dir + 'emu_data.csv'
    mle_loc = results_dir + 'mle_data.csv'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

def set_up_directories(args):
    """
    add doc
    """
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    if not os.path.exists(args.output_dir + run_label):
        os.mkdir(args.output_dir + run_label)

    if not os.path.exists(args.output_dir + run_label + '/figures'):
        os.mkdir(args.output_dir + run_label + '/figures')
    return

def save_eval_params_file(args):
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    with open(args.output_dir + run_label + '/evaluationParameters.json','w') as json_file:
        json.dump(eval_params, json_file, indent=4)

    return

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

def save_eval_params_file(args):
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    run_label = eval_params['run_label']

    with open(args.output_dir + run_label + '/evaluationParameters.json','w') as json_file:
        json.dump(eval_params, json_file, indent=4)

    return

def save_indexed_dataset():
    """
    Save distances separately for specific parameter set.
    Implement later if needed
    """
    return NotImplementedError

def save_indexed_dataset(dataset, index, filename):
    dataset = dataset.iloc[:index]
    save_dataset(dataset, filename)

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

def runtime(seconds):
    hrs = seconds / (60*60)
    roundHrs = int(hrs)
    minutes = hrs - roundHrs
    minutes = minutes * 60
    roundMinutes = round(minutes,2)
    
    return f'{roundHrs} hours {roundMinutes} minutes'

def run_checks(args):
    
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    possible_methods = ['convolution','student-t','gaussian']
    if not (eval_params['stats_distribution_method'] in possible_methods):
        raise(ValueError('Method must be one of the following: \'convolution\',\'student-t\',\'gaussian\''))

    if (args.output_dir[-1] != '/') and (eval_params['emulator_output_folder_path'][-1] != '/'):
        raise(ValueError('End OutputDir and emulator_output_folder_path with \'/\' character'))

    if args.output_dir[-1] != '/':
        raise(ValueError('End OutputDir with \'/\' character'))
    
    if eval_params['emulator_output_folder_path'][-1] != '/':
        raise(ValueError('End emulator_output_folder_path with \'/\' character'))
    
    return