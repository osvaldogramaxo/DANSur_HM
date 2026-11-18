"""
DANSur_HM in bilby_torch's batched likelihood (https://github.com/osvaldogramaxo/bilby_torch):
the log-likelihood ratio of

    bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers,
        bilby.gw.WaveformGenerator(time_domain_source_model=dansur_binary_black_hole, ...),
        priors=priors, time_marginalization=..., distance_marginalization=...,
    ).log_likelihood_ratio()

for many parameter points in one call, in torch on CPU or GPU, optionally marginalised over
the orbital phase; and nessai or pocoMC sampling with it (BatchedNessai, BatchedPocoMC).
Requires bilby_torch (the ``nessai`` or ``pocomc`` extra).

Check against bilby:  python -m dansur.batched_likelihood [cpu|cuda]
"""
import gwtools as _gwtools
import numpy as np
import torch
from bilby_torch import BatchedLikelihood, BatchedNessai, BatchedPocoMC  # noqa: F401 (re-exported)
from gwtools.harmonics import s_lambda_lm

from dansur.surrogate import DANSur


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

    local_max = torch.zeros_like(nonzero)
    local_max[:, 1:-1] = (amp[:, 1:-1] >= amp[:, :-2]) & (amp[:, 1:-1] >= amp[:, 2:])
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
    """The batched waveform of bilby_torch.BatchedLikelihood for a DANSur instance: the
    H_m of dansur_binary_black_hole, placed on the segment as model_gen_func does."""

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
        H = {}
        for (l, m), hlm in modes.items():
            if m <= 0:
                continue
            hlm = _taper_start(hlm).gather(1, roll)
            yp = torch.as_tensor(_sylm(l, m, theta, np.pi / 2), device=dev)[:, None]
            yn = torch.as_tensor((-1) ** l * _sylm(l, -m, theta, np.pi / 2), device=dev)[:, None]
            H[m] = H.get(m, 0) + yp * hlm
            H[-m] = H.get(-m, 0) + yn * hlm.conj()
        return H

    return waveform


def BatchedDANSurLikelihood(likelihood, model=None, device='cpu', **kwargs):
    """bilby_torch.BatchedLikelihood with DANSur_HM as the waveform. likelihood: the
    equivalent bilby GravitationalWaveTransient (with dansur_binary_black_hole); model: a
    DANSur instance, default DANSur(device=device); kwargs as BatchedLikelihood
    (orbital_phase_marginalization, time_upsampling, chunk_size, ...)."""
    model = model if model is not None else DANSur(device=device)
    return BatchedLikelihood(likelihood, dansur_modes(model), device=model.device, **kwargs)


if __name__ == '__main__':
    import sys
    from bilby_torch.selfcheck import run
    from dansur.bilby_source import dansur_binary_black_hole

    device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    model = DANSur(device=device)
    # The float32 network rounds differently for different batch sizes (relative ~1e-5 on
    # some CPU builds, enough to move log L of poor templates by ~0.1), so on CPU evaluate it
    # one row at a time: the comparisons then test everything else to round-off
    if device == 'cpu':
        network = model._sur_dimless

        def one_row_at_a_time(x):
            unique, inverse = torch.unique(x, dim=0, return_inverse=True)
            return torch.cat([network(unique[i:i + 1]) for i in range(len(unique))])[inverse]
        model._sur_dimless = one_row_at_a_time
    run(dansur_binary_black_hole, dansur_modes(model), device=device, tol=1e-8 if device == 'cpu' else 1e-3)
