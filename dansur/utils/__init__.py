"""
DANSur Utilities Package.

This package provides organized submodules for:
- `nn`: Neural network architectures and layers (Decoder, nnPCA, invPCA, resLin, SinActivation).
- `waveforms`: Phase manipulation, overlap/mismatch metrics, and spherical harmonic projections.
- `data`: Datasets (SXSDataset, MyDataset), MultiEpochsDataLoader, normalizers, and dict/HDF5 converters.
- `losses`: Loss functions (HMLoss, SXSLoss, ASDL1Loss, LpLoss, weighted losses).
- `training`: Training loops (train, test), EarlyStopping callback, and polynomial basis functions.
- `generation`: Approximant waveform generation and dataset building (SEOBNRv5HM, NRSur7dq4, IMRPhenomTPHM).
- `helpers`: Path, logging, and plotting utilities.
"""

from .nn import (
    Decoder,
    nnPCA,
    invPCA,
    resLin,
    SinActivation,
)

from .waveforms import (
    unwrap_phase,
    wrap_phase,
    get_phase,
    get_phases,
    torch_unwrap,
    to_wave,
    wave_to_amp_phase,
    get_wave_power,
    torch_overlap,
    np_overlap,
    myoverlap,
    mymismatch,
    latent_mismatch,
    Cover_Sphere,
)

from .data import (
    SXSDataset,
    MyDataset,
    MultiEpochsDataLoader,
    _RepeatSampler,
    UnitGaussianNormalizer,
    GaussianNormalizer,
    RangeNormalizer,
    list_of_dicts_to_dict_of_lists,
    convert_dict_to_list_of_dicts,
    hdf5_to_dict,
)

from .losses import (
    weighted_mse_loss,
    weighted_L1_loss,
    ASDL1Loss,
    HMLoss,
    SXSLoss,
    LpLoss,
)

from .training import (
    EarlyStopping,
    train,
    test,
    legendreDer,
    phi_,
    get_phi_psi,
    get_filter,
)

from .generation import (
    get_dimensionless_approx,
    get_cplx_wave,
    get_hm_wave,
    gen_hm_data,
    generate_dataset,
)

from .helpers import (
    get_stdout_path,
    get_stderr_path,
    get_folder_from_path,
    plot_hist_from_binned_statistic,
)

__all__ = [
    # nn
    'Decoder',
    'nnPCA',
    'invPCA',
    'resLin',
    'SinActivation',
    # waveforms
    'unwrap_phase',
    'wrap_phase',
    'get_phase',
    'get_phases',
    'torch_unwrap',
    'to_wave',
    'wave_to_amp_phase',
    'get_wave_power',
    'torch_overlap',
    'np_overlap',
    'myoverlap',
    'mymismatch',
    'latent_mismatch',
    'Cover_Sphere',
    # data
    'SXSDataset',
    'MyDataset',
    'MultiEpochsDataLoader',
    '_RepeatSampler',
    'UnitGaussianNormalizer',
    'GaussianNormalizer',
    'RangeNormalizer',
    'list_of_dicts_to_dict_of_lists',
    'convert_dict_to_list_of_dicts',
    'hdf5_to_dict',
    # losses
    'weighted_mse_loss',
    'weighted_L1_loss',
    'ASDL1Loss',
    'HMLoss',
    'SXSLoss',
    'LpLoss',
    # training
    'EarlyStopping',
    'train',
    'test',
    'legendreDer',
    'phi_',
    'get_phi_psi',
    'get_filter',
    # generation
    'get_dimensionless_approx',
    'get_cplx_wave',
    'get_hm_wave',
    'gen_hm_data',
    'generate_dataset',
    # helpers
    'get_stdout_path',
    'get_stderr_path',
    'get_folder_from_path',
    'plot_hist_from_binned_statistic',
]
