"""
Code for training emulators to the Smoke PPE and performing parameter inference
"""
import src.confidence
import src.emulator
import src.implausibility
import src.mle
import src.model_discrepancy
import src.storage

__version__ = '0.1.0'

__all__ = [
    'smokeppe_constraints',
    '__version__'
]
