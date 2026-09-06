"""
Waveform processing, phase manipulation, overlap/mismatch metrics, and spherical harmonic projections.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from pyspherical import spin_spherical_harmonic


# ========== Phase Processing ==========

def unwrap_phase(complex_array: np.ndarray) -> np.ndarray:
    """
    Unwrap the phase of a complex array.
    """
    phase = np.angle(complex_array)
    return np.unwrap(phase)


def wrap_phase(phase: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Wrap phase values to [-pi, pi].
    """
    return np.angle(np.exp(1j * phase))


def get_phase(elem: np.ndarray) -> np.ndarray:
    """
    Get zero-aligned, unwrapped phase from a complex array.
    """
    out = unwrap_phase(elem)
    out = out - out[0]
    out = out * np.sign(out.mean())
    return out


def get_phases(array: Union[np.ndarray, torch.Tensor], set_init_zero: bool = True) -> torch.Tensor:
    """
    Unwrap and align phases across multidimensional complex arrays.
    Returns float tensor.
    """
    if isinstance(array, torch.Tensor):
        arr_np = array.cpu().numpy()
    else:
        arr_np = array
    out = unwrap_phase(arr_np)
    if set_init_zero:
        out = out - out[..., np.newaxis, 0]
    out = out * np.sign(out.mean(axis=-1)[..., np.newaxis])
    return torch.as_tensor(out, dtype=torch.float32)


def torch_unwrap(
    p: torch.Tensor,
    discont: Optional[float] = None,
    axis: int = -1,
    period: float = torch.pi
) -> torch.Tensor:
    """
    PyTorch implementation of phase unwrapping along a specified axis.
    """
    p = torch.as_tensor(p)
    nd = p.ndim
    dd = torch.diff(p, dim=axis)
    if discont is None:
        discont = period / 2.0
    slice1 = [slice(None, None)] * nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    interval_high = period / 2.0
    boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = (dd - interval_low) % period + interval_low
    if boundary_ambiguous:
        mask = (ddmod == interval_low) & (dd > 0)
        ddmod[mask] = interval_high
    ph_correct = ddmod - dd
    ph_correct[torch.abs(dd) < discont] = 0
    up = p.clone()
    up[slice1] = p[slice1] + ph_correct.cumsum(dim=axis)
    return up


# ========== Waveform Representation & Conversion ==========

def to_wave(x: Union[np.ndarray, torch.Tensor], plus_cross: bool = False):
    """
    Convert concatenated [amplitude, phase] array to complex waveform or plus/cross polarizations.
    """
    amp = x[:, :x.shape[-1] // 2]
    phase = x[:, x.shape[-1] // 2:]
    if isinstance(x, torch.Tensor):
        phase = (phase - phase[:, 0:1]) * torch.sign(phase[:, -1:].real)
        if plus_cross:
            return amp + 1j * phase
        return amp * torch.exp(1j * phase)
    else:
        phase = (phase - phase[:, 0:1]) * np.sign(np.real(phase[:, -1:]))
        if plus_cross:
            return amp + 1j * phase
        return amp * np.exp(1j * phase)


def wave_to_amp_phase(wave: torch.Tensor) -> torch.Tensor:
    """
    Convert complex waveform tensor to concatenated [amplitude, phase] tensor.
    """
    amp = torch.abs(wave)
    phase = get_phases(wave)
    return torch.cat([amp, phase], dim=-1)


def get_wave_power(x: Union[np.ndarray, torch.Tensor], normalize: bool = True):
    """
    Compute total integrated waveform power.
    """
    w = to_wave(x)
    if isinstance(w, np.ndarray):
        power = np.sum(np.abs(w), axis=-1)
    else:
        power = torch.sum(torch.abs(w), dim=-1)
    if normalize:
        power = power / power.max()
    return power


# ========== Overlap and Mismatch Metrics ==========

def torch_overlap(h1: torch.Tensor, h2: torch.Tensor, dt: float = 2.0, df: Optional[float] = None) -> torch.Tensor:
    """
    Calculate normalized overlap between waveforms using PyTorch in frequency domain.
    Clamped to [0, 1] using Hardtanh.
    """
    h1_f = torch.fft.fft(h1) * dt
    h2_f = torch.fft.fft(h2) * dt
    df_val = 1.0 / len(h1) / dt if df is None else df
    sig_norm = 4.0 * df_val

    sig1 = torch.sqrt((h1_f.conj() * h1_f).sum(axis=-1).real * sig_norm)
    sig2 = torch.sqrt((h2_f.conj() * h2_f).sum(axis=-1).real * sig_norm)

    norm = 1.0 / sig1 / sig2
    inner = (h1_f.conj() * h2_f).sum(axis=-1)
    return nn.Hardtanh()((4.0 * df_val * inner * norm).real)


def np_overlap(h1: np.ndarray, h2: np.ndarray, dt: float = 2.0, df: Optional[float] = None) -> np.ndarray:
    """
    Calculate normalized overlap between waveforms using NumPy in frequency domain.
    Clamped to [0, 1].
    """
    h1_f = np.fft.fft(h1) * dt
    h2_f = np.fft.fft(h2) * dt
    df_val = 1.0 / len(h1) / dt if df is None else df
    sig_norm = 4.0 * df_val

    sig1 = np.sqrt((h1_f.conj() * h1_f).sum(axis=-1).real * sig_norm)
    sig2 = np.sqrt((h2_f.conj() * h2_f).sum(axis=-1).real * sig_norm)

    norm = 1.0 / sig1 / sig2
    inner = (h1_f.conj() * h2_f).sum(axis=-1)
    x = (4.0 * df_val * inner * norm).real
    hardtanh = np.maximum(0.0, np.minimum(1.0, x))
    return hardtanh


def myoverlap(h1: Any, h2: Any, dt: float = 2.0, df: Optional[float] = None):
    """
    Universal overlap calculator supporting both NumPy arrays and PyTorch tensors.
    """
    if isinstance(h1, np.ndarray):
        h2_np = h2.cpu().numpy() if isinstance(h2, torch.Tensor) else h2
        return np_overlap(h1, h2_np, dt, df)
    elif isinstance(h1, torch.Tensor):
        h2_t = torch.as_tensor(h2, device=h1.device) if not isinstance(h2, torch.Tensor) else h2
        return torch_overlap(h1, h2_t, dt, df)
    else:
        raise ValueError('Input must be numpy array or torch tensor')


def mymismatch(h1: torch.Tensor, h2: torch.Tensor, dt: float = 2.0, df: Optional[float] = None) -> torch.Tensor:
    """
    Calculate mismatch (1 - overlap), replacing NaNs with 0.
    """
    ov = myoverlap(h1.double(), h2.double(), dt, df)
    if isinstance(ov, np.ndarray):
        ov = torch.from_numpy(ov)
    return torch.nan_to_num(1.0 - ov, nan=0.0)


def latent_mismatch(amp_phase_1: torch.Tensor, amp_phase_2: torch.Tensor, dt: float = 2.0, df: Optional[float] = None) -> torch.Tensor:
    """
    Calculate mismatch directly from concatenated amplitude-phase representations.
    """
    ov = myoverlap(to_wave(amp_phase_1), to_wave(amp_phase_2), dt, df)
    if isinstance(ov, np.ndarray):
        ov = torch.from_numpy(ov)
    return torch.nan_to_num(1.0 - ov, nan=0.0)


# ========== Spherical Harmonics Projection ==========

class Cover_Sphere(nn.Module):
    """
    Project higher-order modes onto the 2-sphere grid using spin-weighted spherical harmonics.

    Input tensor shape: (batch, n_modes, time_samples)
    Output tensor shape: (batch, n_theta, n_phi, time_samples)
    """
    def __init__(self, mode_map: Dict[int, Tuple[int, int]], n: int = 10, infer_neg_m: bool = False, device: str = 'cpu'):
        super(Cover_Sphere, self).__init__()
        self.n = n
        self.mode_map = dict(mode_map)
        self.infer_neg_m = infer_neg_m
        if infer_neg_m:
            self.neg_m_l_coeffs = torch.tensor([(-1)**l for _, (l, _) in self.mode_map.items()]).to(device)[None, :, None]
            mode_map_neg_m = {k + len(mode_map): (l, -m) for k, (l, m) in self.mode_map.items()}
            self.mode_map = {**self.mode_map, **mode_map_neg_m}

        self.theta = np.linspace(0, np.pi, n + 2)[1:-1]
        self.phi = np.linspace(0, 2.0 * np.pi, n + 1)[:-1]
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)
        self.l_values = np.array([self.mode_map[i][0] for i in range(len(self.mode_map))])
        self.m_values = np.array([self.mode_map[i][1] for i in range(len(self.mode_map))])
        self.sylm_values = torch.from_numpy(
            np.array([
                spin_spherical_harmonic(-2, self.l_values[i], self.m_values[i], self.THETA, self.PHI)
                for i in range(len(self.mode_map))
            ])
        ).to(device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.infer_neg_m:
            neg_m_h = self.neg_m_l_coeffs.to(h.device) * torch.conj(h)
            h = torch.cat([h, neg_m_h], dim=1)

        h_sphere = torch.einsum('ijk,lim->ljkm', self.sylm_values.to(h.device), h)
        return h_sphere
