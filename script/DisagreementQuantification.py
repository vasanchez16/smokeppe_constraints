import numpy as np
import pandas as pd
from ModelDiscrepancy import ModelDiscrepancy
from MLE import MLE
from Implausibilities import Implausibilities
import time
from src.storage.utils import formatRuntime

def DisagreementQuantification(args):

    print('---------DisagreementQuantification---------')

    DQ_start_time = time.time()

    distances, variances = ModelDiscrepancy(args)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: {formatRuntime(timeNow)[0]} hours {formatRuntime(timeNow)[1]} minutes')
    MLE(args, distances, variances)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: {formatRuntime(timeNow)[0]} hours {formatRuntime(timeNow)[1]} minutes')
    Implausibilities(args, distances, variances)
    timeNow = time.time() - DQ_start_time
    print(f'DQ Runtime: {formatRuntime(timeNow)[0]} hours {formatRuntime(timeNow)[1]} minutes')

    return None