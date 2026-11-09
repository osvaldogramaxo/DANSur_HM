"""
bilby source model for DANSur_HM.

Requires the ``bilby`` extra: ``pip install "dansur-hm[bilby]"``.

Example::

    import bilby
    from dansur.bilby_source import dansur_binary_black_hole

    wfg = bilby.gw.WaveformGenerator(
        duration=4, sampling_frequency=2048,
        time_domain_source_model=dansur_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    )
"""
from functools import lru_cache
from typing import Any

import bilby
import numba
import numpy as np
from gwsurrogate.surrogate import _sYlm
from numpy import floating
from numpy.typing import NDArray

from dansur.surrogate import DANSur


@numba.njit
def _numba_taper_complex(data_real, data_imag):
    """Taper both real and imaginary parts of a complex mode in one pass.
    Equivalent to two calls of _numba_taper_start but avoids redundant
    boundary-detection work by reusing the taper window computed for the
    real part on the imaginary part as well.
    Returns (tapered_real, tapered_imag).
    """
    length = len(data_real)
    start = -1
    for i in range(length):
        if data_real[i] != 0.0 or data_imag[i] != 0.0:
            start = i
            break
    if start == -1:
        return data_real, data_imag
    end = -1
    for i in range(length - 1, -1, -1):
        if data_real[i] != 0.0 or data_imag[i] != 0.0:
            end = i
            break
    if end - start <= 1:
        return data_real, data_imag
    mid = (start + end) // 2
    flag = 0
    i = start + 1
    n = 0
    LALSIMULATION_RINGING_EXTENT = 19
    amp_i = (data_real[i]**2 + data_imag[i]**2) ** 0.5
    while flag < 2 and i != mid:
        amp_im1 = (data_real[i-1]**2 + data_imag[i-1]**2) ** 0.5
        amp_ip1 = (data_real[i+1]**2 + data_imag[i+1]**2) ** 0.5
        if amp_i >= amp_im1 and amp_i >= amp_ip1:
            if amp_i == amp_ip1:
                i += 1
                amp_i = amp_ip1
            if i - start > LALSIMULATION_RINGING_EXTENT:
                flag += 1
            n = i - start
        i += 1
        if i < length:
            amp_i = (data_real[i]**2 + data_imag[i]**2) ** 0.5
    if flag < 2:
        n = mid - start
    if n > 1:
        data_real[start] = 0.0
        data_imag[start] = 0.0
        realN = float(n)
        for j in range(start + 1, start + n - 1):
            realI = float(j - start)
            z = (realN - 1.0) / realI + (realN - 1.0) / (realI - (realN - 1.0))
            sigma = 1.0 / (np.exp(z) + 1.0)
            data_real[j] = data_real[j] * sigma
            data_imag[j] = data_imag[j] * sigma
    return data_real, data_imag


@numba.njit(parallel=True)
def _numba_taper_roll_align_modes(
    modes_real,      # shape (n_modes, n_samples) — real parts, C-contiguous, will be modified in-place
    modes_imag,      # shape (n_modes, n_samples) — imag parts
    roll_by,         # int: number of positions to roll left (peak_idx)
    m_vals,          # int64 array, shape (n_modes,): the m index of each mode
    phi22_peak,      # float: arg(h22[0]) after rolling
):
    """Fused kernel: taper + roll + peak-phase align for all modes.

    Operates in-place on modes_real / modes_imag (pre-copied).
    Returns (out_real, out_imag) of the same shape, with:
      1. Planck taper applied at the start of each mode.
      2. Array rolled left by roll_by (cyclic, merger placed at index 0).
      3. Each mode rotated by exp(-i * (m/2) * phi22_peak) to zero the
         (2,2) peak phase and preserve relative inter-mode phases.
    """
    n_modes, n = modes_real.shape
    out_real = np.empty_like(modes_real)
    out_imag = np.empty_like(modes_imag)

    for k in numba.prange(n_modes):  # parallel over modes
        # --- 1. Taper ---
        dr = modes_real[k].copy()
        di = modes_imag[k].copy()
        dr, di = _numba_taper_complex(dr, di)

        # --- 2. Roll left by roll_by (equivalent to np.roll(arr, -roll_by)) ---
        roll_by_mod = roll_by % n
        tmp_r = np.empty(n)
        tmp_i = np.empty(n)
        for j in range(n):
            src = (j + roll_by_mod) % n
            tmp_r[j] = dr[src]
            tmp_i[j] = di[src]

        # --- 3. Phase align: rotate by exp(-i * (m/2) * phi22_peak) ---
        m_half = m_vals[k] / 2.0
        angle = -m_half * phi22_peak
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        for j in range(n):
            r = tmp_r[j]
            im = tmp_i[j]
            out_real[k, j] = r * cos_a - im * sin_a
            out_imag[k, j] = r * sin_a + im * cos_a

    return out_real, out_imag


@numba.njit(parallel=True)
def _numba_mode_sum_from_precomputed(
    modes_real,    # shape (n_modes, n_samples)
    modes_imag,    # shape (n_modes, n_samples)
    ylm_real,      # shape (n_modes,): real part of sYlm(-2, l, m, theta, phi)
    ylm_imag,      # shape (n_modes,): imag part
):
    """Mode sum: h = sum_{l,m in dict} Y_{l,m}(theta, phi) * h_{l,m}

    Replaces gwsurrogate's _mode_sum with fake_neg_modes=False.
    The (-m) modes are already present in the mode array (DANSur.__call__
    inserts them when inclination=None), so no symmetry correction is needed.
    Runs parallel over samples.
    Returns (h_real, h_imag) of shape (n_samples,).
    """
    n_modes, n = modes_real.shape
    h_real = np.zeros(n)
    h_imag = np.zeros(n)

    for j in numba.prange(n):
        hr = 0.0
        hi = 0.0
        for k in range(n_modes):
            ar = ylm_real[k]
            ai = ylm_imag[k]
            mr = modes_real[k, j]
            mi = modes_imag[k, j]
            hr += ar * mr - ai * mi
            hi += ar * mi + ai * mr
        h_real[j] = hr
        h_imag[j] = hi

    return h_real, h_imag


def model_gen_func(model: DANSur, times: NDArray[floating[Any]] | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
    """Generic waveform generator function for DANSur-compatible surrogate models.

    Intended to be partially applied with a specific model instance::

        gen_func = partial(model_gen_func, my_model)

    Parameters
    ----------
    model:
        A loaded :class:`DANSur` (or compatible) surrogate instance.
    times:
        Time array in seconds. If ``None``, a default 3 s grid at 1024 Hz is used.
    **kwargs:
        Bilby-style GW parameters (chirp_mass, mass_ratio, chi_1, chi_2,
        luminosity_distance, theta_jn, phase, geocent_time, ra, dec, psi, …).
        May also contain a ``'parameters'`` key wrapping a sub-dict, which bilby
        sometimes injects.

    Returns
    -------
    dict with keys ``'plus'`` and ``'cross'`` (real float64 arrays).
    """
    if len(kwargs) == 0:
        raise ValueError('model_gen_func received no parameters')

    if times is None:
        duration_local = 3.0
        sampling_frequency_local = 1024.0
        times_arr = np.arange(0, duration_local, 1.0 / sampling_frequency_local)
    else:
        times_arr = times
        dt = times_arr[1] - times_arr[0]
        duration_local = len(times_arr) * dt
        sampling_frequency_local = 1.0 / dt

    if 'parameters' in kwargs:
        kwargs = kwargs['parameters']

    converted_params = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(kwargs)[0]
    out = dansur_convert(times_arr, **converted_params)

    # 1. Generate modes (inclination=None)
    out_no_inc: dict[str, Any] = out.copy()
    out_no_inc['inclination'] = None
    out_no_inc['phi_ref'] = 0.0
    out_no_inc['f_low'] = 0.0

    # ensure mass ratio by convention is 0<q<=1
    out_no_inc['q'] = out_no_inc['q'] if out_no_inc['q'] <= 1 else 1 / out_no_inc['q']

    # Evaluate full surrogate waveform on its innate domain
    out_no_inc['times'] = None
    domain_full, h_modes_full, _ = model(**out_no_inc)
    # Calculate times relative to a fixed merger epoch to avoid GPS time-scale errors.
    # We ignore the sampled geocent_time in kwargs for grid definition.
    post_trig = domain_full[-1]
    t_merger = float(times_arr[0]) + duration_local - post_trig
    times_relative = times_arr - t_merger

    # Interpolate each mode onto the user's times_relative
    h_modes: dict[tuple[int, int], np.ndarray] = {}
    for mode, hlm in h_modes_full.items():
        h_modes[mode] = np.interp(times_relative, domain_full, hlm, left=0, right=0)

    # The PN relative-phase offsets between the modes are already applied by
    # DANSur.__call__; applying them here again would double them.

    # Stack modes into contiguous arrays for Numba kernels.
    # Mode order is fixed by dict insertion order (Python 3.7+).
    mode_list_ordered: list[tuple[int, int]] = list(h_modes.keys())
    n_modes = len(mode_list_ordered)
    n_samples = len(times_arr)
    modes_real = np.empty((n_modes, n_samples), dtype=np.float64)
    modes_imag = np.empty((n_modes, n_samples), dtype=np.float64)
    m_vals = np.empty(n_modes, dtype=np.int64)
    for k, (l, m) in enumerate(mode_list_ordered):
        modes_real[k] = h_modes[(l, m)].real
        modes_imag[k] = h_modes[(l, m)].imag
        m_vals[k] = m

    # peak_idx: index of merger in times_relative (closest to t=0)
    peak_idx: int = int(np.argmin(np.abs(times_relative)))

    # Fused kernel: taper + cyclic roll, without the peak-phase rotation. The
    # modes of DANSur.__call__ already have a well-defined frame (phases fixed at
    # the start of the window); rotating by exp(-i*(m/2)*arg h22(peak)), with
    # arg wrapped to (-pi, pi], flips the sign of the odd-m modes wherever the
    # peak phase crosses +-pi, i.e. many times across parameter space.
    out_r, out_i = _numba_taper_roll_align_modes(
        modes_real.copy(), modes_imag.copy(),
        peak_idx, m_vals, 0.0
    )

    # Perform the mode sum: h = sum_{l,m} Y_{l,m}(theta, phi) * h_{l,m}.
    # DANSur.__call__ with inclination=None already inserts (l,-m) modes into
    # the returned dict via equatorial symmetry, so h_modes contains both +m
    # and -m entries. We therefore use fake_neg_modes=False (plain sum), which
    # is what the original _mode_sum call did. sYlm scalars are constant for
    # fixed (theta_jn, phi_ref) and precomputed here to stay out of Numba.
    theta_jn = float(out['inclination'])
    phase = float(out['phi_ref'])
    phi_eval = float(np.pi / 2.0 - phase)  # LAL convention
    ylm_r = np.empty(n_modes)
    ylm_i = np.empty(n_modes)
    for k, (l, m) in enumerate(mode_list_ordered):
        y = _sYlm(-2, l, m, theta_jn, phi_eval)
        ylm_r[k] = y.real
        ylm_i[k] = y.imag

    h_real, h_imag = _numba_mode_sum_from_precomputed(
        out_r, out_i,
        ylm_r, ylm_i,
    )

    # h = h+ - i hx (LAL convention), so hx = -Im(h).
    return {"plus": h_real, "cross": -h_imag}


def dansur_convert(
    times,
    **kwargs) -> dict[str, Any]:
    out_pars = dict(
        q=kwargs['mass_ratio'],
        chiA0=kwargs['chi_1'],
        chiB0=kwargs['chi_2'],
        M=kwargs['total_mass'],
        dist_mpc=kwargs['luminosity_distance'],
        f_low=0,
        times=times,
        f_ref=None,
        freqs=None,
        inclination=kwargs['theta_jn'],
        phi_ref=kwargs['phase'],
        units='mks',
    )
    return out_pars


@lru_cache(maxsize=None)
def _default_model() -> DANSur:
    return DANSur()


def dansur_binary_black_hole(
    time_array, mass_1, mass_2, luminosity_distance, chi_1, chi_2, theta_jn, phase, **kwargs
):
    """bilby time-domain source model using the bundled DANSur_HM model (CPU).

    Masses in solar masses (detector frame), distance in Mpc, aligned spins
    chi_1, chi_2. Returns ``{'plus', 'cross'}`` with the merger at the start of
    the segment (bilby's time-domain convention). To use a different model
    instance, call :func:`model_gen_func` directly.
    """
    return model_gen_func(
        _default_model(), time_array,
        mass_ratio=mass_2 / mass_1, total_mass=mass_1 + mass_2,
        chi_1=chi_1, chi_2=chi_2, luminosity_distance=luminosity_distance,
        theta_jn=theta_jn, phase=phase,
    )
