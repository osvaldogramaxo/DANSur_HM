"""
Legacy compatibility module for common utilities.

Re-exports common utilities from their respective specialized submodules.
"""

from .waveforms import (
    unwrap_phase,
    wrap_phase,
    get_phase,
    torch_unwrap,
    torch_overlap,
    np_overlap,
    myoverlap,
    mymismatch,
    latent_mismatch,
)

from .losses import (
    weighted_mse_loss,
    weighted_L1_loss,
)

from .data import (
    MyDataset,
    MultiEpochsDataLoader,
    _RepeatSampler,
    list_of_dicts_to_dict_of_lists,
)

from .nn import (
    SinActivation,
    nnPCA,
    invPCA,
    resLin,
)

from .helpers import (
    get_stdout_path,
    get_stderr_path,
    get_folder_from_path,
)

__all__ = [
    'unwrap_phase',
    'wrap_phase',
    'get_phase',
    'torch_unwrap',
    'torch_overlap',
    'np_overlap',
    'myoverlap',
    'mymismatch',
    'latent_mismatch',
    'weighted_mse_loss',
    'weighted_L1_loss',
    'MyDataset',
    'MultiEpochsDataLoader',
    '_RepeatSampler',
    'list_of_dicts_to_dict_of_lists',
    'SinActivation',
    'nnPCA',
    'invPCA',
    'resLin',
    'get_stdout_path',
    'get_stderr_path',
    'get_folder_from_path',
]
