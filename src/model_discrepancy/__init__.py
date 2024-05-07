"""This folder will be used for the Maximum Likelihood Estimation and the 
quantificaiton/estimation of Implausibility values.
"""
from .model_discrepancy import model_discrepancy
from .utils import calculate_distances_and_variances

__all__ = [
    'model_discrepancy',
    'calculate_distances_and_variances'
]
