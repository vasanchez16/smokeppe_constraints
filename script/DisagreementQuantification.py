import numpy as np
import pandas as pd
from ModelDiscrepancy import ModelDiscrepancy
from MLE import MLE
from Implausibilities import Implausibilities
import time

def DisagreementQuantification(args):

    print('---------DisagreementQuantification---------')

    DQ_start_time = time.time()
    runtime_hrs_lambda = lambda t: float(t / (60*60))

    distances, variances = ModelDiscrepancy(args)
    print(f'DQ Runtime: {runtime_hrs_lambda(time.time()-DQ_start_time)} hours')
    MLE(args, distances, variances)
    print(f'DQ Runtime: {runtime_hrs_lambda(time.time()-DQ_start_time)} hours')
    Implausibilities(args, distances, variances)
    print(f'DQ Runtime: {runtime_hrs_lambda(time.time()-DQ_start_time)} hours')

    return None