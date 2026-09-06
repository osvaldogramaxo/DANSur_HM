"""
Legacy compatibility module for SXS utilities.

Re-exports SXS-related classes and functions from `dansur.utils.data`, `dansur.utils.losses`, and `dansur.utils.training`.
"""

from .data import (
    SXSDataset,
    UnitGaussianNormalizer,
    GaussianNormalizer,
    RangeNormalizer,
    list_of_dicts_to_dict_of_lists,
)
from .losses import (
    SXSLoss,
    LpLoss,
)
from .training import (
    legendreDer,
    phi_,
    get_phi_psi,
    get_filter,
    train,
    test,
)

__all__ = [
    'SXSDataset',
    'UnitGaussianNormalizer',
    'GaussianNormalizer',
    'RangeNormalizer',
    'list_of_dicts_to_dict_of_lists',
    'SXSLoss',
    'LpLoss',
    'legendreDer',
    'phi_',
    'get_phi_psi',
    'get_filter',
    'train',
    'test',
]
