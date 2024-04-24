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

def runtime(seconds):
    hrs = int(seconds / (60*60))
    minutes = int((seconds % (60*60)) / 60)
    return f'Current Runtime: {runtime(timeNow)[0]} hours {runtime(timeNow)[1]}\
        minutes'
