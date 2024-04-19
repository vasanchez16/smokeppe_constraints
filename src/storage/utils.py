import pandas as pd
import os


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


def save_dataset(dataset, filename):
    dataset.to_csv(filename, index=True)


def save_indexed_dataset(dataset, index, filename):
    dataset = dataset.iloc[:index]
    save_dataset(dataset, filename)

def formatRuntime(seconds):
    hrs = seconds / (60*60)
    roundHrs = int(hrs)
    minutes = hrs - roundHrs
    minutes = minutes * 60
    roundMinutes = round(minutes,2)
    
    return [roundHrs , roundMinutes]