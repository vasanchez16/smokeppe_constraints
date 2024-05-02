import numpy as np
import pandas as pd
from ModelDiscrepancy import ModelDiscrepancy
from MLE import MLE
from Implausibilities import Implausibilities
import time

from src.storage.utils import runtime

def DisagreementQuantification(args):

    print('---------DisagreementQuantification---------')

    DQ_start_time = time.time()

    distances, variances = ModelDiscrepancy(args)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: ' + runtime(timeNow))
    MLE(args, distances, variances)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: ' + runtime(timeNow))
    Implausibilities(args, distances, variances)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: ' + runtime(timeNow))

    return None