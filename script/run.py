import argparse
import configparser
import time
import sys
import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from src.model_discrepancy.model_discrepancy import model_discrepancy
from src.mle.mle import mle
from src.implausibility.implausibility import implausibilities
from src.confidence.frequentist_confidence_set import frequentist_confidence_set
from src.confidence.viz import variant_distribution_comp
from src.storage.utils import (runtime,
                               set_up_directories,
                               run_checks,
                               save_eval_params_file)

# Config
config = configparser.ConfigParser()
config.read('config.ini')

input_file = config.get('DEFAULT', 'InputFile')
output_dir = config.get('DEFAULT', 'OutputDir')


def main(args):
    """
    Main function to run the simulation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    input_file
    output_dir

    Returns:
        None
    """
    start_time = time.time()

    # Set-up
    run_checks(args)
    set_up_directories(args)
    save_eval_params_file(args)
    with open(args.input_file,'r') as file:
        eval_params = json.load(file)
    stats_dist_method = eval_params['stats_distribution_method']

    run_label = eval_params['run_label']
    print(f'Run label: {run_label}')
    
    """
    2. Estimate model discrepancy
    """
    distances, variances = model_discrepancy(args)
    print(runtime(time.time() - start_time))

    """
    3. Compute MLE
    """
    mle(args, distances, variances)
    print(runtime(time.time() - start_time))

    """
    4. Compute implausibilities
    """
    implausibilities(args, distances, variances)
    if stats_dist_method == 'student-t_bootstrap':
        variant_distribution_comp(args, distances, variances)
    print(runtime(time.time() - start_time))

    """
    5. Compute confidence sets
    """
    frequentist_confidence_set(args, distances, variances)
    print(runtime(time.time() - start_time))

    """
    Runtime report
    """
    print('job successful')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument("--savefigs", action="store_true", default=False)
    parser.add_argument("--laplace", action="store_true", default=False)
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
