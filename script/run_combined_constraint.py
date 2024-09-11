import argparse
import configparser
import time
import sys
import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from src.implausibility.combined_implausibility import combined_implausibilities
from src.confidence.combined_frequentist_confidence_set import combined_frequentist_confidence_set
from src.storage.utils import (runtime,
                               set_up_directories_combined_implaus,
                               run_checks,
                               save_eval_params_file)

# Config
config = configparser.ConfigParser()
config.read('config.ini')

input_file = config.get('DEFAULT', 'InputFile')
output_dir = config.get('DEFAULT', 'OutputDir')


def main(args):
    """
    Function for combining constraints of multiple target variables.

    Args:
        args (argparse.Namespace): Command-line arguments.

    input_file
    output_dir

    Returns:
        None
    """
    start_time = time.time()

    # Set-up
    set_up_directories_combined_implaus(args)
    save_eval_params_file(args)
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)

    run_dirs = eval_params['directories']

    """
    1. Calculate combined implausibility values
    """
    combined_implausibilities(args, run_dirs)
    print(runtime(time.time() - start_time))

    """
    2. Visualize implausibilities
    """
    combined_frequentist_confidence_set(args)
    print(runtime(time.time() - start_time))

    """
    Runtime report
    """
    print('job successful')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=input_file
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir
    )
    args = parser.parse_args()
    main(args)
