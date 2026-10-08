#%%
from argparse import ArgumentParser
from bilby.gw.prior import BBHPriorDict

from gwtools import Msuninsec
import os
import argparse
import ast
from typing import Any
from glob import glob
import numpy as np
from numpy._typing._array_like import NDArray
from numpy._typing._nbit import _NBitIntP
from numpy import floating, intp, signedinteger
import h5py
import lal
import bilby
import bilby.core.prior as bcp
import bilby.gw.prior as bgp
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.core.prior.analytical import Uniform, Cosine, Sine
from bilby.core.prior.base import Constraint
from astropy.cosmology import LambdaCDM
from matplotlib import pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import lalsimulation as lalsim
import torch

from dansur.utils.utils import *
from dansur.surrogate.sur_utils import DANSur
from functools import wraps
from gwsurrogate.surrogate import _sYlm
import numba


@numba.njit
def _numba_taper_start(data):
    length = len(data)
    start = -1
    for i in range(length):
        if data[i] != 0.0:
            start = i
            break
    if start == -1:
        return data
    end = -1
    for i in range(length - 1, -1, -1):
        if data[i] != 0.0:
            end = i
            break
    if end - start <= 1:
        return data
    mid = (start + end) // 2
    flag = 0
    i = start + 1
    n = 0
    LALSIMULATION_RINGING_EXTENT = 19
    while flag < 2 and i != mid:
        val_i = abs(data[i])
        if val_i >= abs(data[i-1]) and val_i >= abs(data[i+1]):
            if val_i == abs(data[i+1]):
                i += 1
            if i - start > LALSIMULATION_RINGING_EXTENT:
                flag += 1
            n = i - start
        i += 1
    if flag < 2:
        n = mid - start
    if n > 1:
        data[start] = 0.0
        realN = float(n)
        for j in range(start + 1, start + n - 1):
            realI = float(j - start)
            z = (realN - 1.0) / realI + (realN - 1.0) / (realI - (realN - 1.0))
            sigma = 1.0 / (np.exp(z) + 1.0)
            data[j] = data[j] * sigma
    return data


@numba.njit
def _numba_compute_v0(phase_0: float, phase_1: float, dt: float) -> float:
    """Compute initial characteristic velocity v0 = (omega_orb)^(1/3)
    from two consecutive (2,2)-mode phase samples.
    
    omega_22 = d(phi_22)/dt  =>  omega_orb = omega_22 / 2
    v0 = omega_orb^(1/3)
    
    Takes the absolute value of omega_22 to guard against sign conventions.
    Returns a real, positive v0.
    """
    omega_22 = (phase_1 - phase_0) / dt
    if omega_22 < 0.0:
        omega_22 = -omega_22
    omega_orb = omega_22 / 2.0
    # cube root via exponentiation
    return omega_orb ** (1.0 / 3.0)


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

    # Re-apply PN relative-phase offsets between modes.
    # The surrogate network predicts mode amplitudes and phases independently;
    # their relative phases must be consistent with PN theory at early times
    # (inspiral). We compute the initial characteristic velocity v0 from the
    # (2,2) phase gradient on the full (pre-interpolation) surrogate domain,
    # then rotate each mode h_lm -> h_lm * exp(i * xi_lm) where
    # xi_lm = arg(H_lm^PN(v0)) is the PN complex-amplitude phase.
    # The (2,2) mode is the gauge reference (xi_22 = 0 by definition).
    # This step must happen BEFORE the peak-phase alignment below, because
    # that alignment applies a mode-dependent rotation exp(-i * m/2 * phi22_peak)
    # which *preserves* relative phases under the coorbital-frame convention.
    h22_full = h_modes_full[(2, 2)]
    if len(h22_full) >= 2:
        dt_full = float(domain_full[1] - domain_full[0])
        phi0 = float(np.angle(h22_full[0]))
        phi1 = float(np.angle(h22_full[1]))
        v0_pn = _numba_compute_v0(phi0, phi1, dt_full)
    else:
        v0_pn = 0.0

    q_val    = float(out_no_inc['q'])
    chi1_val = float(out_no_inc['chiA0']) if np.ndim(out_no_inc['chiA0']) == 0 else float(out_no_inc['chiA0'][-1])
    chi2_val = float(out_no_inc['chiB0']) if np.ndim(out_no_inc['chiB0']) == 0 else float(out_no_inc['chiB0'][-1])

    # Ensure q >= 1 for get_pn_amplitude (it uses q = m1/m2 >= 1)
    q_pn = q_val if q_val >= 1.0 else 1.0 / q_val

    if v0_pn > 0.0:
        xi_lm: dict[tuple[int, int], float] = {}
        for (l, m) in h_modes.keys():
            if l == 2 and m == 2:
                xi_lm[(l, m)] = 0.0
            else:
                H_lm_pn = model.get_pn_amplitude(l, m, v0_pn, q_pn, chi1_val, chi2_val)
                xi_lm[(l, m)] = float(np.angle(H_lm_pn)) if H_lm_pn != 0j else 0.0
        h_modes = {
            mode: hlm * np.exp(1j * xi_lm[mode])
            for mode, hlm in h_modes.items()
        }

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

    # Need phi22_peak to pass to the fused kernel, but it depends on the
    # h22 value AFTER taper+roll. We do a quick pre-roll on h22 only to get
    # phi22_peak, then pass it into the full fused kernel.
    # This is cheaper than splitting into separate passes.
    h22_idx = next(i for i, (l, m) in enumerate(mode_list_ordered) if l == 2 and m == 2)
    h22_r_tmp = modes_real[h22_idx].copy()
    h22_i_tmp = modes_imag[h22_idx].copy()
    h22_r_tmp, h22_i_tmp = _numba_taper_complex(h22_r_tmp, h22_i_tmp)
    n = n_samples
    roll_by_mod = peak_idx % n
    h22_r_peak = h22_r_tmp[roll_by_mod]
    h22_i_peak = h22_i_tmp[roll_by_mod]
    phi22_peak: float = float(np.arctan2(h22_i_peak, h22_r_peak))

    # Fused kernel: taper + cyclic roll + phase alignment for all modes.
    # Phase alignment rotates mode (l,m) by exp(-i*(m/2)*phi22_peak),
    # zeroing the (2,2) peak phase while preserving relative inter-mode phases.
    out_r, out_i = _numba_taper_roll_align_modes(
        modes_real.copy(), modes_imag.copy(),
        peak_idx, m_vals, phi22_peak
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

    return {"plus": h_real, "cross": h_imag}


# Setup paths relative to the script's directory for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if "LAL_DATA_PATH" not in os.environ:
    os.environ["LAL_DATA_PATH"] = os.path.join(PROJECT_ROOT, ".venv/share/lalsimulation")

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

def find_pe_key0(f: h5py.File) -> str:
    """Finds the best top-level key containing priors/analytic inside an HDF5 PE file."""
    for k in f.keys():
        if isinstance(f[k], h5py.Group) and 'priors' in f[k] and 'analytic' in f[k]['priors']:
            return k
    for k in f.keys():
        if isinstance(f[k], h5py.Group) and 'priors' in f[k]:
            return k
    return list(f.keys())[0]

def is_single_number(val: Any) -> bool:
    if isinstance(val, (int, float, np.number)):
        return True
    if isinstance(val, bytes):
        try:
            val = val.decode('utf-8')
        except Exception:
            return False
    if isinstance(val, str):
        try:
            float(val)
            return True
        except ValueError:
            return False
    return False

def make_plotting_callback(outdir: str, active_wf_gen: Any, lal_wf_gen: Any, dansur_wf_gen: Any, sxs_inj_wf_gen: Any, priors: bilby.gw.prior.BBHPriorDict, approx: str, injection_parameters: dict) -> Any:
    import os
    import json
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for cluster runs
    import matplotlib.pyplot as plt

    def plotting_callback(sampler: Any) -> None:
        try:
            iteration = sampler.iteration
            
            # Find the highest likelihood point in the current live points
            max_idx: intp = np.argmax(sampler.live_points['logL'])
            best_point: Any = sampler.live_points[max_idx]
            
            # Build the parameter dictionary (start with defaults/fixed, overwrite sampled)
            best_params: dict[str, Any] = priors.sample()
            for name in sampler.model.names:
                best_params[name] = float(best_point[name])
            
            # Ensure directory exists
            waveforms_dir = os.path.join(outdir, 'waveforms')
            os.makedirs(waveforms_dir, exist_ok=True)
            
            # --- 1. Save auxiliary JSON file with ALL parameters ---
            serializable_params: dict[str, Any] = {}
            for k, v in best_params.items():
                if isinstance(v, (np.floating, np.integer)):
                    serializable_params[k] = v.item()
                elif isinstance(v, (float, int, str)):
                    serializable_params[k] = v
                else:
                    serializable_params[k] = str(v)
            
            serializable_params['log_likelihood'] = float(best_point['logL'])
            serializable_params['iteration'] = int(iteration)

            json_path = os.path.join(waveforms_dir, f'waveform_it{iteration}.json')
            with open(json_path, 'w') as f:
                json.dump(serializable_params, f, indent=4)
            
            # --- 2. Plot the waveform with key parameters in the title ---
            plt.figure(figsize=(10, 6))
            try:
                # Plot reference LAL waveform
                lal_strain: np.ndarray = lal_wf_gen.time_domain_strain(best_params)['plus']
                plt.plot(lal_wf_gen.time_array - lal_wf_gen.time_array[0], lal_strain, 'k', label='LAL')
                
                # Plot true SXS injection waveform
                inj_strain: np.ndarray = sxs_inj_wf_gen.time_domain_strain(injection_parameters)['plus']
                plt.plot(sxs_inj_wf_gen.time_array - sxs_inj_wf_gen.time_array[0], inj_strain, 'g:', label='True SXS Inj', alpha=0.8)
                
                # Plot DANSur waveform if applicable
                if approx == "DANSur":
                    dansur_strain: np.ndarray = active_wf_gen.time_domain_strain(best_params)['plus']
                    plt.plot(active_wf_gen.time_array - active_wf_gen.time_array[0], dansur_strain, '--', label='DANSur')
                elif approx == "NRHybSur3dq8_gwsurrogate":
                    dansur_strain = dansur_wf_gen.time_domain_strain(best_params)['plus']
                    plt.plot(dansur_wf_gen.time_array - dansur_wf_gen.time_array[0], dansur_strain, '--', label='DANSur')
                    gwsur_strain: np.ndarray = active_wf_gen.time_domain_strain(best_params)['plus']
                    plt.plot(active_wf_gen.time_array - active_wf_gen.time_array[0], gwsur_strain, '-.', label='NRHybSur3dq8_gwsurrogate')
                    
            except Exception as e:
                print(f"Error plotting waveform at iteration {iteration}: {e}")
                plt.close()
                return

            key_params: dict[str, Any] = {
                r"$\mathcal{M}_c$": best_params.get('chirp_mass'),
                "$q$": best_params.get('mass_ratio'),
                r"$\chi_1$": best_params.get('chi_1'),
                r"$\chi_2$": best_params.get('chi_2'),
                "$d_L$": best_params.get('luminosity_distance'),
                r"$\theta_{JN}$": best_params.get('theta_jn'),
                r"$\psi$": best_params.get('psi'),
                
            }
            param_parts: list[str] = [f"{k}={v:.3f}" for k, v in key_params.items() if v is not None]
            param_str: str = ", ".join(param_parts)

            plt.title(f"Iteration {iteration} - logL: {best_point['logL']:.2f}\n{param_str}", fontsize=10)
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")
            plt.legend()
            plt.savefig(os.path.join(waveforms_dir, f'waveform_it{iteration}.png'))
            plt.close()

            # --- 3. Save sampler checkpoint so we can resume ---
            import pickle
            from nessai.utils import safe_file_dump
            safe_file_dump(
                sampler, sampler.resume_file, pickle, save_existing=True
            )
        except Exception as e:
            print(f"Error in plotting callback at iteration {sampler.iteration}: {e}")
        
    return plotting_callback

class SXSInjection:
    def __init__(self, sxs_id, h5_path=None):
        # Determine the file path
        if h5_path is None:
            paths = ["data/sxs_waveforms.h5", "data/sxs_waves_4modes_named.h5"]
            for p in paths:
                full_p = os.path.join(PROJECT_ROOT, p)
                if os.path.exists(full_p):
                    with h5py.File(full_p, "r") as f:
                        if sxs_id in f:
                            h5_path = full_p
                            break
            if h5_path is None:
                raise ValueError(f"SXS ID {sxs_id} not found in standard H5 files.")
        
        self.sxs_id = sxs_id
        self.h5_path = h5_path
        
        with h5py.File(self.h5_path, "r") as f:
            group: h5py.Group = f[self.sxs_id]
            assert isinstance(group, h5py.Group)
            self.metadata = {k: v[()] for k, v in group["metadata"].items()}
            self.ts_metadata = {k: v[()] for k, v in group["TS_metadata"].items() if not isinstance(v, h5py.Group)}
            
            self.t_dim = self.ts_metadata["time"]
            
            self.modes_data = {}
            waveform = group["waveform"]
            assert isinstance(waveform, h5py.Group)
            for mode_str in waveform.keys():
                mode_tuple = ast.literal_eval(mode_str)
                # wf_mode
                self.modes_data[mode_tuple] = waveform[mode_str][()]
        
        if (2, 2) not in self.modes_data:
            raise ValueError("SXS waveform must contain the (2, 2) mode.")
        
        h_22 = self.modes_data[(2, 2)]
        self.peak_idx = np.argmax(np.abs(h_22))
        self.t_peak_dim = self.t_dim[self.peak_idx]
        self.t_dim_shifted = self.t_dim - self.t_peak_dim
        
        self.q_sxs = float(self.metadata["mass_ratio"])
        spin1 = self.metadata["dimensionless_spin1"]
        spin2 = self.metadata["dimensionless_spin2"]
        self.chi1z = float(spin1[-1])
        self.chi2z = float(spin2[-1])

def mode_sum(h_modes, theta, phi):
    h = 0.
    for (ell, m), h_mode in h_modes.items():
        h += _sYlm(-2, ell, m, theta, phi) * h_mode
    return h

def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Run parameter estimation on SXS waveform injections")
    parser.add_argument("--sxs-id", type=str, default="SXS:BBH:0069", help="SXS ID to inject")
    parser.add_argument("--event", type=str, default=None, help="The optional GW event name to process for PSD only")
    parser.add_argument("--modes", type=str, default=None, help="Modes to use. Example: '[(2,2), (3,3)]'")
    parser.add_argument("--approx", type=str, default="DANSur", help="The approximant to use (e.g. DANSur, NRHybSur3dq8, NRSur7dq4)")
    parser.add_argument("--resume", action="store_true", help="Resume the sampling process")
    parser.add_argument("--sampler", type=str, default="nessai", help="The sampler to use (e.g. (i)nessai, dynesty, cpnest)")
    parser.add_argument("--npoints", type=int, default=1000, help="The number of live points for the sampler")
    parser.add_argument("--min-samples", type=int, default=1000, help="The minimum number of samples for the sampler")
    parser.add_argument("--stopping", type=float, default=0.1, help="The stopping criterion threshold for the sampler")
    parser.add_argument("--time-marginalization", action="store_true", help="Activate time marginalization in the likelihood")
    parser.add_argument("--zero-noise", action="store_true", help="Inject signal into zero-noise")
    parser.add_argument("--narrow-prior", action="store_true", help="Use a narrow prior centered around true injection values")
    
    # Injection parameters overrides
    parser.add_argument("--inj-total-mass", type=float, default=80.0, help="Injection total mass (Msun)")
    parser.add_argument("--inj-distance", type=float, default=500.0, help="Injection luminosity distance (Mpc)")
    parser.add_argument("--inj-theta-jn", type=float, default=np.pi/2, help="Injection theta_jn (rad)")
    parser.add_argument("--inj-phase", type=float, default=1.5, help="Injection phase (rad)")
    parser.add_argument("--inj-psi", type=float, default=0.8, help="Injection polarization angle psi (rad)")
    parser.add_argument("--inj-ra", type=float, default=1.3, help="Injection right ascension ra (rad)")
    parser.add_argument("--inj-dec", type=float, default=-0.4, help="Injection declination dec (rad)")
    parser.add_argument("--inj-geocent-time", type=float, default=1126259642.413, help="Injection GPS geocenter time")
    
    args, _ = parser.parse_known_args()
    
    duration: float = 4
    sampling_frequency: int = 1024*2
    outdir: str = f"bilby/outdir_{args.approx}_{args.sxs_id.replace(':', '_')}"
    DEFAULT_MODES: list[tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]

    if args.modes:
        modes: list[tuple[int, int]] | None = ast.literal_eval(args.modes)
        if modes is not None:
            outdir += str(modes)
        else:
            modes = DEFAULT_MODES
    else:
        modes = DEFAULT_MODES

    priors: BBHPriorDict = bilby.gw.prior.BBHPriorDict(aligned_spin=True)

    label = args.approx
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # 1. Load the SXS waveform to inject
    sxs_inj = SXSInjection(args.sxs_id)
    
    # Intrinsic parameters from SXS
    q_sxs = sxs_inj.q_sxs if sxs_inj.q_sxs <= 1.0 else 1.0 / sxs_inj.q_sxs
    chi1_sxs = sxs_inj.chi1z # heavier body
    chi2_sxs = sxs_inj.chi2z # lighter body (chi_2 in mass_ratio >= 1 prior space)
    
    # Calculate chirp mass
    eta = q_sxs / (1.0 + q_sxs)**2
    total_mass_inj = args.inj_total_mass
    chirp_mass_inj = total_mass_inj * (eta)**0.6
    
    injection_parameters = dict(
        chirp_mass=chirp_mass_inj,
        mass_ratio=q_sxs,
        chi_1=chi1_sxs,
        chi_2=chi2_sxs,
        luminosity_distance=args.inj_distance,
        theta_jn=args.inj_theta_jn,
        phase=args.inj_phase,
        psi=args.inj_psi,
        ra=args.inj_ra,
        dec=args.inj_dec,
        geocent_time=args.inj_geocent_time,
    )
    
    print("\n" + "="*60)
    print("SXS WAVEFORM INJECTION PARAMETERS:")
    for k, v in injection_parameters.items():
        print(f"  {k:20s}: {v}")
    print("="*60 + "\n")

    # 2. Set up the injection source model
    def sxs_inj_gen_func(times=None, **kwargs):
        if 'parameters' in kwargs:
            kwargs = kwargs['parameters']
        converted_params = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(kwargs)[0]
        
        M_tot = converted_params['total_mass']
        dist_mpc = converted_params['luminosity_distance']
        theta_jn = converted_params['theta_jn']
        phase = converted_params['phase']
        
        # Scale dimensionless time to physical time
        t_scale = M_tot * Msuninsec
        t_phys_shifted = sxs_inj.t_dim_shifted * t_scale
        
        # Physical strain amplitude scaling
        G_val = 6.6743e-11
        c_val = 299792458.0
        M_sun_val = 1.9884e30
        mpc_in_m = 3.085677581491367e+22
        amp_scale = (G_val * M_tot * M_sun_val) / (c_val**2 * dist_mpc * mpc_in_m)
        
        # Determine merger time relative to the times array
        post_trig_dim = sxs_inj.t_dim_shifted[-1]
        post_trig_phys = post_trig_dim * t_scale
        t_merger = float(times[0]) + duration - post_trig_phys
        times_relative = times - t_merger
        
        # Interpolate modes
        h_modes = {}
        for mode, hlm in sxs_inj.modes_data.items():
            h_modes[mode] = np.interp(times_relative, t_phys_shifted, hlm, left=0, right=0)
            
        # Taper
        h_modes_tapered = {}
        for mode, hlm in h_modes.items():
            r_tapered = _numba_taper_start(hlm.real.copy())
            i_tapered = _numba_taper_start(hlm.imag.copy())
            h_modes_tapered[mode] = r_tapered + 1j * i_tapered
            
        # Roll
        peak_idx = np.argmin(np.abs(times_relative))
        h_modes_rolled = {mode: np.roll(h_modes_tapered[mode], -peak_idx) for mode in h_modes_tapered}
        
        # Phase alignment to match conventions
        h22_peak_phase = np.angle(h_modes_rolled[(2, 2)][0])
        h_modes_aligned = {
            (l, m): hlm * np.exp(-1j * (m / 2.0) * h22_peak_phase)
            for (l, m), hlm in h_modes_rolled.items()
        }
        
        # Apply amplitude scale
        h_modes_scaled = {mode: hlm * amp_scale for mode, hlm in h_modes_aligned.items()}
        
        # Mode sum
        h_cycled = mode_sum(h_modes_scaled, theta_jn, np.pi/2 - phase)
        return {"plus": h_cycled.real, "cross": h_cycled.imag}

    # Custom source model key list
    targ_keys = set(bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters=priors.sample())[0].keys())

    sxs_inj_wf_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        time_domain_source_model=sxs_inj_gen_func,
    )
    sxs_inj_wf_generator.source_parameter_keys = targ_keys

    # 3. Set up the recovery waveform generator
    waveform_arguments = dict(
        waveform_approximant=args.approx if args.approx not in ["DANSur", "NRHybSur3dq8_gwsurrogate"] else "NRHybSur3dq8",
        minimum_frequency=10.0,
        reference_frequency=10.0
    )

    lal_wf_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    model_path = os.path.join(PROJECT_ROOT, 'models/rolling/DANSur_HM_jitted.pt')
    dansur = DANSur(path=model_path, modes_list=modes)

    # dansur_gen_func is a thin wrapper around model_gen_func with `dansur` pre-bound.
    # We use a wrapper (not bare functools.partial) because bilby's WaveformGenerator
    # inspects __name__ and __doc__ on the source model, which partial objects lack.
    @wraps(model_gen_func)
    def dansur_gen_func(times=None, **kwargs):
        return model_gen_func(dansur, times, **kwargs)

    dansur_wf_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        time_domain_source_model=dansur_gen_func,
    )
    dansur_wf_generator.source_parameter_keys = targ_keys

    if args.approx == "NRHybSur3dq8_gwsurrogate":
        import gwsurrogate
        sur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
        assert isinstance(sur, gwsurrogate.surrogate.NRHybSur3dq8)
        
        def gwsur_gen_func(times=None, **kwargs):
            if times is None:
                duration_local = duration
                sampling_frequency_local = sampling_frequency
                times_arr = np.arange(0, duration_local, 1.0 / sampling_frequency_local)
            else:
                times_arr = times
                dt = times_arr[1] - times_arr[0]
                duration_local = len(times_arr) * dt
                sampling_frequency_local = 1.0 / dt
            converted_params = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(kwargs)[0]
            out = dansur_convert(times_arr, **converted_params)
            
            out_no_inc = out.copy()
            out_no_inc['q'] = 1/out_no_inc['q']
            out_no_inc['inclination'] = None
            out_no_inc['phi_ref'] = 0.0
            out_no_inc['f_low']=0.0
            out_no_inc['mode_list'] = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]
            
            chi1 = out_no_inc['chiA0']
            chi2 = out_no_inc['chiB0']
            if np.ndim(chi1) == 0:
                out_no_inc['chiA0'] = [0.0, 0.0, float(chi1)]
                out_no_inc['chiB0'] = [0.0, 0.0, float(chi2)]
            else:
                out_no_inc['chiA0'] = np.stack([np.zeros_like(chi1), np.zeros_like(chi1), chi1], axis=-1)
                out_no_inc['chiB0'] = np.stack([np.zeros_like(chi2), np.zeros_like(chi2), chi2], axis=-1)
            
            m_val = out_no_inc.get('M')
            t_scale = Msuninsec * float(m_val if m_val is not None else 0.0)
            t_max_physical = sur._sur_dimless.domain[-1] * t_scale
            
            post_trig = t_max_physical
            t_merger = float(times_arr[0]) + duration_local - post_trig
            times_relative = times_arr - t_merger
            
            out_no_inc['times'] = times_relative
            _, h_modes, _ = sur(**out_no_inc)
            
            h_22 = h_modes[(2, 2)]
            peak_idx = np.argmax(np.abs(h_22))
            
            h_modes_tapered = {}
            for mode, hlm in h_modes.items():
                r_tapered = _numba_taper_start(hlm.real.copy())
                i_tapered = _numba_taper_start(hlm.imag.copy())
                h_modes_tapered[mode] = r_tapered + 1j * i_tapered
                
            h_modes_rolled = {mode: np.roll(h_modes_tapered[mode], -peak_idx) for mode in h_modes_tapered}
            
            theta_jn = out['inclination']
            phase = out['phi_ref']
            h_cycled = sur._mode_sum(h_modes_rolled, theta_jn, np.pi/2 - phase, fake_neg_modes=False)
            return {"plus": h_cycled.real, "cross": h_cycled.imag}
            
        gwsur_wf_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters, 
            time_domain_source_model=gwsur_gen_func,
        )
        gwsur_wf_generator.source_parameter_keys = targ_keys

    # 4. Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    h1, l1, v1 = ifos

    gps = args.inj_geocent_time
    start_time = gps - duration + 2.0  # standard merger location
    
    # Set PSDs
    if args.event:
        pe_file = glob(os.path.join(PROJECT_ROOT, f'event_ts/{args.event}/*PEDataRelease*'))[0]
        with h5py.File(pe_file, 'r') as f:
            key0 = find_pe_key0(f)
            f_key0 = f[key0]
            assert isinstance(f_key0, h5py.Group), f"Expected Group, got {type(f_key0)}"
            psds = f_key0['psds']
            assert isinstance(psds, h5py.Group), f"Expected Group, got {type(psds)}"
            
            if 'H1' in psds:
                h1_ds = psds['H1']
                assert isinstance(h1_ds, h5py.Dataset), f"Expected Dataset, got {type(h1_ds)}"
                h1_psd = FrequencySeries(h1_ds[:,1], frequencies=h1_ds[:,0])
                h1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(h1_psd.frequencies, h1_psd)
            else:
                h1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
                
            if 'L1' in psds:
                l1_ds = psds['L1']
                assert isinstance(l1_ds, h5py.Dataset), f"Expected Dataset, got {type(l1_ds)}"
                l1_psd = FrequencySeries(l1_ds[:,1], frequencies=l1_ds[:,0])
                l1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(l1_psd.frequencies, l1_psd)
            else:
                l1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
                
            if 'V1' in psds:
                v1_ds = psds['V1']
                assert isinstance(v1_ds, h5py.Dataset), f"Expected Dataset, got {type(v1_ds)}"
                v1_psd = FrequencySeries(v1_ds[:,1], frequencies=v1_ds[:,0])
                v1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(v1_psd.frequencies, v1_psd)
            else:
                v1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(psd_file='AdV_psd.txt')
    else:
        h1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        l1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        v1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(psd_file='AdV_psd.txt')

    # 5. Generate noise and inject the SXS signal
    for ifo in ifos:
        if args.zero_noise:
            ifo.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time,
            )
        else:
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time,
            )

    # Inject SXS waveform signal
    ifos.inject_signal(
        waveform_generator=sxs_inj_wf_generator,
        parameters=injection_parameters,
    )

    # 6. Define priors
    if args.narrow_prior:
        print("Using NARROW priors around injection values...")
        priors['chirp_mass'] = bilby.core.prior.Uniform(
            minimum=chirp_mass_inj * 0.98,
            maximum=chirp_mass_inj * 1.02,
            name='chirp_mass',
            latex_label='$\\mathcal{M}$'
        )
        priors['mass_ratio'] = bilby.core.prior.Uniform(
            minimum=max(1.0, q_sxs * 0.9),
            maximum=min(8.0, q_sxs * 1.1),
            name='mass_ratio',
            latex_label='$q$'
        )
        priors['mass_1'] = bilby.gw.prior.Constraint(minimum=5, maximum=200, name='mass_1', latex_label='$m_1$')
        priors['mass_2'] = bilby.gw.prior.Constraint(minimum=5, maximum=200, name='mass_2', latex_label='$m_2$')

        priors['chi_1'] = bilby.core.prior.Uniform(
            minimum=max(-0.8, chi1_sxs - 0.1),
            maximum=min(0.8, chi1_sxs + 0.1),
            name='chi_1',
            latex_label='$\\chi_1$'
        )
        priors['chi_2'] = bilby.core.prior.Uniform(
            minimum=max(-0.8, chi2_sxs - 0.1),
            maximum=min(0.8, chi2_sxs + 0.1),
            name='chi_2',
            latex_label='$\\chi_2$'
        )
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            minimum=args.inj_distance * 0.8,
            maximum=args.inj_distance * 1.2,
            name='luminosity_distance',
            latex_label='$d_L$'
        )
        priors['geocent_time'] = bilby.core.prior.Uniform(
            minimum=gps - 0.05,
            maximum=gps + 0.05,
            name='geocent_time',
            latex_label='$t_c$'
        )
        priors['theta_jn'] = bilby.core.prior.Uniform(
            minimum=max(0.0, args.inj_theta_jn - 0.2),
            maximum=min(np.pi, args.inj_theta_jn + 0.2),
            name='theta_jn',
            latex_label='$\\theta_{JN}$'
        )
        priors['phase'] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=2.0 * np.pi,
            name='phase',
            latex_label='$\\phi_c$',
            boundary='periodic'
        )
        priors['psi'] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=np.pi,
            name='psi',
            latex_label='$\\psi$',
            boundary='periodic'
        )
        priors['ra'] = bilby.core.prior.Uniform(
            minimum=max(0.0, args.inj_ra - 0.2),
            maximum=min(2.0*np.pi, args.inj_ra + 0.2),
            name='ra',
            latex_label='$ra$'
        )
        priors['dec'] = bilby.core.prior.Uniform(
            minimum=max(-np.pi/2, args.inj_dec - 0.2),
            maximum=min(np.pi/2, args.inj_dec + 0.2),
            name='dec',
            latex_label='$dec$'
        )
    else:
        print("Using WIDE BBH priors...")
        priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=5, maximum=150, name='chirp_mass', latex_label='$\\mathcal{M}$')
        priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(minimum=1/8, maximum=1, name='mass_ratio', latex_label='$q$')
        priors['mass_1'] = bilby.gw.prior.Constraint(minimum=5, maximum=200, name='mass_1', latex_label='$m_1$')
        priors['mass_2'] = bilby.gw.prior.Constraint(minimum=5, maximum=200, name='mass_2', latex_label='$m_2$')

        priors['chi_1'] = bilby.gw.prior.AlignedSpin(
            a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.8), 
            z_prior=bilby.core.prior.Uniform(minimum=-1, maximum=1), 
            name='chi_1', latex_label='$\\chi_1$', minimum=-0.8, maximum=0.8
        )
        priors['chi_2'] = bilby.gw.prior.AlignedSpin(
            a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.8), 
            z_prior=bilby.core.prior.Uniform(minimum=-1, maximum=1), 
            name='chi_2', latex_label='$\\chi_2$', minimum=-0.8, maximum=0.8
        )
        cosmology: LambdaCDM = LambdaCDM(H0=67.9, Om0=0.3065, Ode0=0.6935, Tcmb0=0., Neff=3.04, m_nu=None, Ob0=None) # type: ignore
        priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(
            minimum=10.0, maximum=2000.0, cosmology=cosmology, name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None
        )
        priors['geocent_time'] = bilby.gw.prior.Uniform(gps - 0.1, gps + 0.1, latex_label="$t_c$")

    if not args.resume:
        print(f"Removing directory {outdir}")
        import shutil
        try:
            shutil.rmtree(outdir)
        except Exception:
            pass

    if args.approx == "DANSur":
        active_wf_gen = dansur_wf_generator 
    elif args.approx == "NRHybSur3dq8_gwsurrogate":
        active_wf_gen = gwsur_wf_generator
    else:
        active_wf_gen = lal_wf_generator

    likelihood = GravitationalWaveTransient(
        interferometers=[x for x in ifos if not ((x.time_domain_strain == 0).all() or (np.isnan(x.time_domain_strain)).all())],
        waveform_generator=active_wf_gen, priors=priors, distance_marginalization=False, phase_marginalization=False, time_marginalization=args.time_marginalization, jitter_time=False
    )

    sampler_kwargs: dict[str, Any] = {}
    if args.sampler == "nessai" or args.sampler == "inessai":
        if args.narrow_prior:
            sampler_kwargs["flow_class"] = "flowproposal"
        else:
            sampler_kwargs["flow_class"] = "gwflowproposal"
        sampler_kwargs["npoints"] = args.npoints
        sampler_kwargs["min_samples"] = args.min_samples
        sampler_kwargs["stopping"] = args.stopping
        
        sampler_kwargs["checkpointing"] = True
        sampler_kwargs["checkpoint_on_iteration"] = True
        sampler_kwargs["checkpoint_interval"] = 1000
        sampler_kwargs["checkpoint_callback"] = make_plotting_callback(
            outdir=outdir,
            active_wf_gen=active_wf_gen,
            lal_wf_gen=lal_wf_generator,
            dansur_wf_gen=dansur_wf_generator,
            sxs_inj_wf_gen=sxs_inj_wf_generator,
            priors=priors,
            approx=args.approx,
            injection_parameters=injection_parameters
        )
    else:
        sampler_kwargs["nlive"] = args.npoints
        if args.sampler == "dynesty":
            sampler_kwargs["dlogz"] = args.stopping

    # Save injection parameters to a JSON file in outdir
    import json
    os.makedirs(outdir, exist_ok=True)
    injection_file = os.path.join(outdir, "injection_parameters.json")
    with open(injection_file, "w") as f:
        json.dump(injection_parameters, f, indent=4)
    print(f"Saved injection parameters to {injection_file}")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=args.sampler,
        resume=args.resume,
        outdir=outdir,
        label=label,
        npool=80,
        **sampler_kwargs
    )

    plot_dict = {
        k: injection_parameters[k]
        for k in injection_parameters.keys()
        if k in result.posterior.columns and np.max(result.posterior[k]) > np.min(result.posterior[k])
    }
    result.plot_corner(parameters=plot_dict)

    plt.figure()
    maxll_params = dict(result.posterior[result.posterior.log_likelihood == result.posterior.log_likelihood.max()].iloc[0])

    plt.plot(lal_wf_generator.time_array - lal_wf_generator.time_array[0], (lal_wf_generator.time_domain_strain(maxll_params)['plus']), 'k', label='LAL MaxL')
    plt.plot(sxs_inj_wf_generator.time_array - sxs_inj_wf_generator.time_array[0], (sxs_inj_wf_generator.time_domain_strain(injection_parameters)['plus']), 'g:', label='True SXS Inj')
    if args.approx == "DANSur":
        plt.plot(dansur_wf_generator.time_array - dansur_wf_generator.time_array[0], (dansur_wf_generator.time_domain_strain(maxll_params)['plus']), '--', label='DANSur MaxL')
    elif args.approx == "NRHybSur3dq8_gwsurrogate":
        plt.plot(gwsur_wf_generator.time_array - gwsur_wf_generator.time_array[0], (gwsur_wf_generator.time_domain_strain(maxll_params)['plus']), '--', label='NRHybSur3dq8 MaxL')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'wfs_end.png'))

    # Print summary of recovered vs true parameters
    print("\n" + "="*60)
    print("PARAMETER RECOVERY SUMMARY:")
    print(f"{'Parameter':20s} | {'True Value':15s} | {'Recovered (Max L)':15s}")
    print("-"*60)
    for k in injection_parameters.keys():
        true_v = injection_parameters[k]
        rec_v = maxll_params.get(k, np.nan)
        print(f"{k:20s} | {true_v:15.5f} | {rec_v:15.5f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
