"""
DANSur_HM's batched waveform, for likelihoods that take many parameter points per call (e.g.
bilby_torch.BatchedLikelihood, https://github.com/osvaldogramaxo/bilby_torch):
dansur_modes(model) returns, for a batch of points, the modes H_m of the bilby source model
dansur_binary_black_hole, placed on the data segment as bilby_source.model_gen_func places them,
in torch on CPU or GPU. With DANSur(hybridize=True) they are continued before the network's
~4000 M with the IMRPhenomT(HM) inspiral.
"""
import gwtools as _gwtools
import numpy as np
import torch
from gwtools.harmonics import s_lambda_lm



def _sylm(l, m, theta, phi, s=-2):
    """gwtools.harmonics.sYlm (used by gwsurrogate) for arrays of theta and phi."""
    sign, ss, mm = 1.0, s, m
    if abs(mm) < abs(ss):
        ss, mm = mm, ss
        if (mm + ss) % 2:
            sign = -sign
    if mm < 0:
        ss, mm = -ss, -mm
        if (mm + ss) % 2:
            sign = -sign
    r = sign * s_lambda_lm(ss, l, mm, np.cos(theta))
    return r * np.cos(m * phi) + 1j * (r * np.sin(m * phi))


def _taper_start(h):
    """Rows of complex h (n, T) with the start taper of bilby_source._numba_taper_complex:
    a Planck-like taper from the first nonzero sample up to the second maximum of the real part
    more than 19 samples in, or up to mid-signal if there is none."""
    T = h.shape[-1]
    idx = torch.arange(T, device=h.device)
    re, im = h.real, h.imag
    amp = re  # maxima of the oscillating real part, as LAL's taper of the real strain
    nonzero = (re != 0) | (im != 0)
    start = torch.where(nonzero, idx, T).amin(dim=1)
    end = torch.where(nonzero, idx, -1).amax(dim=1)
    mid = (start + end) // 2
    off = idx - start[:, None]

    # A maximum is >= its left neighbour and > its right one: a two-sample plateau then counts once,
    # at its second sample, as in LAL's taper and _numba_taper_complex (which step over the twin)
    local_max = torch.zeros_like(nonzero)
    local_max[:, 1:-1] = (amp[:, 1:-1] >= amp[:, :-2]) & (amp[:, 1:-1] > amp[:, 2:])
    qualifying = local_max & (off > 19) & (idx <= mid[:, None] - 1)
    second = qualifying.cumsum(dim=1) >= 2
    n = torch.where(second.any(dim=1), second.int().argmax(dim=1) - start, mid - start)

    apply = (end - start > 1) & (n > 1)  # also False for all-zero rows (end = -1)
    region = apply[:, None] & (off >= 1) & (off <= (n - 2)[:, None])
    n1 = (n - 1).double()[:, None]
    k = torch.where(region, off.double(), 0.5)
    sigma = 1.0 / (torch.exp(n1 / k + n1 / (k - n1)) + 1.0)
    w = torch.where(region, sigma, 1.0)
    w = torch.where(apply[:, None] & (off == 0), 0.0, w)
    return h * w


def dansur_modes(model):
    """The batched waveform for a DANSur instance: a function (p, times) -> {m: H_m} giving the H_m
    of dansur_binary_black_hole for the parameter points p (arrays of mass_1, mass_2, chi_1,
    chi_2, luminosity_distance, theta_jn), placed on the segment as model_gen_func does."""

    @torch.inference_mode()
    def waveform(p, times):
        dev = model.device
        n = len(p['mass_1'])

        # DANSur modes in SI units, as dansur_binary_black_hole -> DANSur.__call__
        M = p['mass_1'] + p['mass_2']
        q = 1.0 / torch.as_tensor(p['mass_2'] / p['mass_1'])  # __call__ inverts q <= 1
        zeros = torch.zeros(n, dtype=torch.float64)
        chi_a = torch.stack([zeros, zeros, torch.as_tensor(p['chi_1'])], -1)
        chi_b = torch.stack([zeros, zeros, torch.as_tensor(p['chi_2'])], -1)
        x = model._get_intrinsic_parameters(q, chi_a, chi_b, None, None, None)
        amp_scale = M * _gwtools.Msuninsec * _gwtools.c / (1e6 * p['luminosity_distance'] * _gwtools.PC_SI)
        t_scale = _gwtools.Msuninsec * M
        # Place the merger as model_gen_func does: on the sample n0 where the end of the network
        # grid falls within dt of the end of the segment, the modes evaluated on the segment
        # (continued with the inspiral before the network grid), the start tapered and the merger
        # rolled to t = 0. The merger sits exactly on a sample, so no sub-sample offset is left
        # between it and geocent_time.
        t_scale = torch.as_tensor(t_scale)
        T, dt = len(times), float(times[1] - times[0])
        k = torch.arange(T, device=dev)
        n0 = torch.round(T - model.t_end * t_scale.to(dev) / dt).long().clamp(0, T - 1)
        times_relative = (k[None, :] - n0[:, None]) * dt
        _, modes = model._evaluate_chunk(x, q, t_scale, torch.as_tensor(amp_scale), None, 0.0, times_relative, None)
        roll = (k[None, :] + n0[:, None]) % T

        # H_m = sum_l Ylm(theta_jn, pi/2) h_lm. For aligned spins h_{l,-m} = (-1)^l conj(h_lm),
        # which placement and taper preserve, so only m > 0 modes are placed
        theta = p['theta_jn']
        pos = [(l, m) for l, m in modes if m > 0]
        if len(pos) * n * T <= 2 ** 17:
            # tiny batches: taper all modes in one call (row-wise, so identical), saving per-op overhead;
            # for larger ones the bigger temporaries cost more than that saves
            tapered = _taper_start(torch.cat([modes[lm] for lm in pos])).gather(1, roll.repeat(len(pos), 1))
            placed = dict(zip(pos, tapered.split(n)))
        else:
            placed = {lm: _taper_start(modes[lm]).gather(1, roll) for lm in pos}
        H = {}
        for (l, m), hlm in placed.items():
            yp = torch.as_tensor(_sylm(l, m, theta, np.pi / 2), device=dev)[:, None]
            yn = torch.as_tensor((-1) ** l * _sylm(l, -m, theta, np.pi / 2), device=dev)[:, None]
            H[m] = H.get(m, 0) + yp * hlm
            H[-m] = H.get(-m, 0) + yn * hlm.conj()
        return H

    return waveform
