"""The folder here will be used for the training of emulators and removal of 
outliers through emulator evaluation.
"""
from .emulator import (Emulator, get_implausibility_from_least_squares_variant)
from .observer import (Observer)
from .simulator import (SimulatedDataset)

__all__ = [
    'Emulator',
    'get_implausibility_from_least_squares_variant',
    'Observer',
    'SimulatedDataset',
    'emulator',
    'get_implausibility_from_least_squares_variant',
    'observer',
    'simulator'
]
