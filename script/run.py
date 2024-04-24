import argparse
import configparser
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from DisagreementQuantification import DisagreementQuantification
from FreqConfSet import FreqConfSet

from src.emulator.evaluate import evaluate
from src.model_discrepancy.model_discrepancy import model_discrepancy
from src.inference.mle import mle
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

    """
    1. Evaluate the emulator
    """
    evaluate(args)
    print(runtime(time.time() - start_time))

    """
    2-3. Estimate the noise model
    """
    distances, variances = model_discrepancy(args)
    print(runtime(time.time() - start_time))

    mle(args, distances, variances)
    print(runtime(time.time() - start_time))

    """
    4. Compute implausibilities
    """
    Implausibilities(args, distances, variances)
    print(runtime(time.time() - start_time))

    """
    5. Compute confidence sets
    """
    FreqConfSet(args)
    print(runtime(time.time() - start_time))

    """
    Runtime report
    """
    timeNow = time.time() - start_time
    print(runtime(timeNow))
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
