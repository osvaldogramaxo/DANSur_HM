"""
Dataset, DataLoader, normalizers, and data format utilities for DANSur.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ========== Dictionary and HDF5 Helpers ==========

def list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Convert a list of dictionaries to a dictionary of lists.
    """
    if not list_of_dicts:
        return {}
    dict_of_lists = {}
    for key in list_of_dicts[0].keys():
        dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
    return dict_of_lists


def convert_dict_to_list_of_dicts(input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a dictionary whose values are array-like to a list of dictionaries per element.
    """
    keys = input_dict.keys()
    values_list = zip(*input_dict.values())
    return [dict(zip(keys, vals)) for vals in values_list]


def hdf5_to_dict(hdf5_obj: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Any:
    """
    Recursively converts an h5py File, Group, or Dataset into a Python dictionary.
    """
    if isinstance(hdf5_obj, (h5py.Group, h5py.File)):
        d = {key: hdf5_to_dict(val) for key, val in hdf5_obj.items()}
        if hdf5_obj.attrs:
            d['_attrs'] = {
                k: (v.decode('utf-8') if isinstance(v, bytes) else v)
                for k, v in hdf5_obj.attrs.items()
            }
        return d
    elif isinstance(hdf5_obj, h5py.Dataset):
        if hdf5_obj.dtype.names is not None:
            data = hdf5_obj[()]
            return {
                name: (data[name].tolist() if isinstance(data[name], np.ndarray) else data[name])
                for name in hdf5_obj.dtype.names
            }
        val = hdf5_obj[()]
        if isinstance(val, bytes):
            return val.decode('utf-8')
        elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
            return [x.decode('utf-8') for x in val]
        return val
    else:
        return hdf5_obj


# ========== Datasets and DataLoaders ==========

class MyDataset(Dataset):
    """
    Generic PyTorch Dataset for features X and targets y.
    """
    def __init__(self, X, y, device='cpu'):
        self.X = torch.as_tensor(X).to(device)
        self.y = torch.as_tensor(y).to(device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class _RepeatSampler(object):
    """
    Sampler that repeats indefinitely to allow reusing DataLoader worker processes.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(DataLoader):
    """
    DataLoader that reuses worker processes across multiple epochs to prevent process recreation overhead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class SXSDataset(Dataset):
    """
    PyTorch Dataset for SXS (Simulating eXtreme Spacetimes) numerical relativity waveform data.
    Loads waveforms, mass ratio q, dimensionless spins, and effective precessing spin chi_p.
    """
    def __init__(self, file_path: str, modes: Optional[List[Tuple[int, int]]] = None):
        self.file_path = file_path

        with h5py.File(file_path, 'r') as file:
            self.data_len = len(file)
            self.waveforms = list(file.keys())
            if modes is None:
                group = file[self.waveforms[0]]
                waveform_group = group['waveform']
                self.modes = np.array(list(waveform_group.keys()))
            else:
                self.modes = [str(x).replace(' ', '') for x in modes]

            waveform_data_list = []
            metadata_list = []
            TS_metadata_list = []
            for i in range(self.data_len):
                key = self.waveforms[i]
                data, metadata_key, TS_metadata_key = self.get_hdf5_item(file, key)
                waveform_data_list.append(data)
                metadata_list.append(metadata_key)
                TS_metadata_list.append(TS_metadata_key)

        self.metadata = list_of_dicts_to_dict_of_lists(metadata_list)
        self.TS_metadata = list_of_dicts_to_dict_of_lists(TS_metadata_list)
        self.waveform_data = np.array(waveform_data_list)

        self.spin1 = np.array(self.metadata['dimensionless_spin1'])
        self.spin2 = np.array(self.metadata['dimensionless_spin2'])
        self.q = np.array(self.metadata['mass_ratio'])
        self.q = np.maximum(self.q, 1.0)
        self.q = 1.0 / self.q
        self.chi_p = self.compute_chi_p(self.q, self.metadata['chi1_perp'], self.metadata['chi2_perp'])
        self.params_data = np.stack([self.q, self.spin1[..., -1], self.spin2[..., -1]], axis=1)

    def get_hdf5_item(self, file: h5py.File, key: str) -> Tuple[np.ndarray, Dict, Dict]:
        group = file[key]
        waveform_group = group['waveform']
        data = np.stack([np.asarray(waveform_group[lm]) for lm in self.modes])

        metadata_group = group['metadata']
        metadata = {k: v[()] for k, v in metadata_group.items()}

        ts_metadata_group = group['TS_metadata']
        TS_metadata = {k: v[()] for k, v in ts_metadata_group.items() if not isinstance(v, h5py.Group)}
        return data, metadata, TS_metadata

    @staticmethod
    def compute_chi_p(q: np.ndarray, S1_perp: Union[np.ndarray, float], S2_perp: Union[np.ndarray, float]) -> np.ndarray:
        """
        Calculate the effective precessing spin parameter chi_p.
        """
        A1 = 2.0 + 3.0 * q / 2.0
        A2 = 2.0 + 3.0 / (2.0 * q)
        return np.maximum(A1 * S1_perp, A2 * S2_perp)

    chi_p = compute_chi_p

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.params_data[idx], dtype=torch.float64),
            torch.tensor(self.waveform_data[idx], dtype=torch.float64)
        )


# ========== Normalizers ==========

class UnitGaussianNormalizer(object):
    """
    Pointwise Gaussian normalization: (x - mean) / (std + eps).
    """
    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx=None) -> torch.Tensor:
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            else:
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        return (x * std) + mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self


class GaussianNormalizer(object):
    """
    Global Gaussian normalization across all elements.
    """
    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx=None) -> torch.Tensor:
        return (x * (self.std + self.eps)) + self.mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self


class RangeNormalizer(object):
    """
    Linear range normalization mapping to [low, high].
    """
    def __init__(self, x: torch.Tensor, low: float = 0.0, high: float = 1.0):
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)
        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        return x.view(s)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        return x.view(s)
