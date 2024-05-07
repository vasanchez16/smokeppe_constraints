"""This folder will be used for the Maximum Likelihood Estimation and the 
quantificaiton/estimation of Implausibility values.
"""
from .gauss import mle_gauss
from .mle import mle
from .student_t import mle_t

__all__ = [
    'mle_gauss',
    'mle',
    'mle_t'
]
