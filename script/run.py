import argparse
import configparser
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from EmulatorEval import EmulatorEval
from DisagreementQuantification import DisagreementQuantification
from FreqConfSet import FreqConfSet
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

    """
    Some checks for code to work properly.
    """
    run_checks(args)
    
    """
    Set up directories
    """
    set_up_directories(args)

    """
    Save eval_params json file
    """
    save_eval_params_file(args)

    """
    Evaluate the emulator
    """
    EmulatorEval(args)
    print(runtime(time.time() - start_time))

    """
    Compute implausibilities
    """
    DisagreementQuantification(args)
    print(runtime(time.time() - start_time))

    """
    Compute confidence sets
    """
    FreqConfSet(args)

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
