"""
Waveform generation and dataset creation from approximants (SEOBNRv5HM, NRSur7dq4, IMRPhenomTPHM).
"""

from typing import Any, Dict, List, Tuple
import h5py
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .waveforms import get_phase
from .data import convert_dict_to_list_of_dicts


def get_dimensionless_approx(q: float, s1: float, s2: float, approximant: str = 'IMRPhenomTPHM') -> Dict:
    """
    Generate dimensionless time-domain modes using LAL waveform generators via gwsignal and Astropy.
    """
    from astropy.constants import G, c
    import astropy.units as u
    from gwpy.timeseries import TimeSeries as gwpy_ts
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    M_sun_s = ((u.solMass.to(u.kg) * u.kg / (c**3)) * G)
    M_sun_m = ((u.solMass.to(u.kg) * u.kg / (c**2)) * G)

    mtot = 50
    mpc_in_m = u.Mpc.to(u.m)

    python_dict = {
        'mass1': m1 * mtot * u.solMass,
        'mass2': m2 * mtot * u.solMass,
        'spin1x': 0 * u.dimensionless_unscaled,
        'spin1y': 0 * u.dimensionless_unscaled,
        'spin1z': s1 * u.dimensionless_unscaled,
        'spin2x': 0 * u.dimensionless_unscaled,
        'spin2y': 0 * u.dimensionless_unscaled,
        'spin2z': s2 * u.dimensionless_unscaled,
        'deltaT': u.s / 4096,
        'f22_start': 10.0 * u.Hz,
        'f22_ref': 10.0 * u.Hz,
        'distance': 1.0 * u.Mpc / mpc_in_m,
        'inclination': np.pi / 2.0 * 0 * u.rad,
        'phi_ref': 0 * u.rad,
        'eccentricity': 0.0 * u.dimensionless_unscaled,
        'longAscNodes': 0.0 * u.rad,
        'meanPerAno': 0.0 * u.rad,
        'condition': 0
    }

    gen = gwsignal_get_waveform_generator(approximant)
    try:
        hlm = gen.generate_td_modes(**python_dict)
    except ValueError:
        hlm = gen.generate_fd_modes(**python_dict)
        modes = list(hlm.keys())[1:]
        for mode in modes:
            fd_data = hlm[mode]
            td_data = np.fft.ifft(4096 * fd_data)
            td_data = np.roll(td_data, len(td_data) // 2)
            times = np.arange(len(td_data)) * python_dict['deltaT'].value
            hlm[mode] = gwpy_ts(td_data, times=times)

    hlm_22_argmax = abs(hlm[2, 2]).argmax()
    modes = list(hlm.keys())[1:]

    for mode in modes:
        hlm[mode].times = hlm[mode].times - hlm[mode].times[hlm_22_argmax]
        hlm[mode].times = hlm[mode].times / (mtot * M_sun_s)

        interp = np.interp(np.arange(-4096 + 100, 100, 2), hlm[mode].times, hlm[mode].value)
        phase = np.angle(interp) - np.angle(interp[0])
        hlm[mode] = gwpy_ts(abs(interp) * np.exp(-1j * phase), times=np.arange(-4096 + 100, 100, 2)) / (M_sun_m.value * mtot)
    return hlm


def get_cplx_wave(param: Dict[str, Any], length: int = 2048, roll: int = 0, whiten: bool = False, sur: str = None, dt: float = 2.0):
    """
    Generate complex (2,2) waveform from a specified surrogate or approximant.
    """
    import gwsurrogate
    import pyseobnr

    qs = 1.0 / param['mass_ratio'] if param['mass_ratio'].max() < 1 else param['mass_ratio']
    if 'Sur' in sur:
        sur_obj = gwsurrogate.LoadSurrogate(sur)
        f_low = 0.004 if 'Hyb' in sur_obj.name else 0.0
        _, h, _ = sur_obj(qs, [0, 0, param['chi_1']], [0, 0, param['chi_2']], times=np.arange(-4096, 125, dt), f_low=f_low, mode_list=[(2, 2)])
        wave_cplx = h[(2, 2)]
    elif ('Phenom' in sur) or ('_PA' in sur):
        h = get_dimensionless_approx(param['mass_ratio'], param['chi_1'], param['chi_2'], sur)
        return h[(2, 2)]
    elif 'SEOBNRv5HM' in sur:
        times, hlm = pyseobnr.generate_waveform.generate_modes_opt(qs, param['chi_1'], param['chi_2'], 1.0 / (70 - qs**1.25), settings={'return_modes': [(2, 2)]})
        wave_cplx = hlm['2,2']
        ap_22_argmax = np.argmax(abs(wave_cplx))
        ap_wf_times = times - times[ap_22_argmax]
        ap_interp = np.interp(np.arange(-4096 + 100, 100, 2), ap_wf_times, wave_cplx)
        ap_interp = abs(ap_interp) * np.exp(1j * get_phase(ap_interp))
        return ap_interp
    else:
        raise ValueError(f'Surrogate model not supported: {sur}')

    amax = np.argmax(abs(wave_cplx))
    wave_cplx = wave_cplx[amax - int((4096 + 100) // dt): amax + int(100 // dt)]
    if roll != 0:
        postmerg_frac = 1.0 / 5.0
        return np.roll(wave_cplx, -int(length * postmerg_frac), axis=-1)[-length:]
    return wave_cplx


def get_hm_wave(
    param: Dict[str, Any],
    modes: List[Tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4)],
    length: int = 2048,
    roll: int = 0,
    whiten: bool = False,
    sur: str = None,
    dt: float = 2.0
) -> np.ndarray:
    """
    Generate multiple complex modes from a surrogate or approximant.
    """
    import gwsurrogate
    import pyseobnr

    qs = 1.0 / param['mass_ratio'] if np.max(param['mass_ratio']) < 1 else param['mass_ratio']
    if 'Sur' in sur:
        sur_obj = gwsurrogate.LoadSurrogate(sur)
        f_low = 0.00 if 'Hyb' in sur_obj.name else 0.0
        _, h, _ = sur_obj(qs, [0, 0, param['chi_1']], [0, 0, param['chi_2']], times=np.arange(-4096 + 100, 100, dt), f_low=f_low, mode_list=modes)
        hproc = np.array([h[mode] for mode in modes], dtype=np.complex128)
        return hproc
    elif ('Phenom' in sur) or ('_PA' in sur):
        h = get_dimensionless_approx(param['mass_ratio'], param['chi_1'], param['chi_2'], sur)
        hproc = np.array([h[mode].value for mode in modes], dtype=np.complex128)
        return hproc
    elif 'SEOBNRv5HM' in sur:
        times, hlm = pyseobnr.generate_waveform.generate_modes_opt(qs, param['chi_1'], param['chi_2'], 1.0 / (70 - qs**1.25), settings={'return_modes': modes})
        wave_cplx = np.array([hlm[str(mode).replace(' ', '').replace('(', '').replace(')', '')] for mode in modes], dtype=np.complex128)
        ap_22_argmax = np.argmax(abs(wave_cplx[0]))
        ap_wf_times = times - times[ap_22_argmax]
        ap_interp = [np.interp(np.arange(-4096 + 100, 100, 2), ap_wf_times, wave_cplx[i]) for i in range(wave_cplx.shape[0])]
        ap_interp = [abs(ap_interp[i]) * np.exp(1j * get_phase(ap_interp[i])) for i in range(wave_cplx.shape[0])]
        return np.stack(ap_interp)
    else:
        raise ValueError(f'Surrogate model not supported: {sur}')


def gen_hm_data(
    inj_params: List[Dict[str, Any]],
    N: int = 1024,
    parallel: bool = False,
    use_tqdm: bool = True,
    whiten: bool = False,
    sur: str = None,
    modes: List[Tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4)],
    dt: float = 2.0
) -> np.ndarray:
    """
    Generate higher-order mode waveform data for a batch of parameters.
    """
    import bilby
    bilby.utils.logging.disable()

    auxfunc = tqdm if use_tqdm else (lambda x: x)
    if parallel:
        out = Parallel(n_jobs=-1)(
            delayed(get_hm_wave)(param, whiten=False, sur=sur, modes=modes, dt=dt)
            for param in auxfunc(inj_params)
        )
    else:
        out = [get_hm_wave(param, whiten=False, sur=sur, modes=modes, dt=dt) for param in auxfunc(inj_params)]
    return np.stack(out)


def generate_dataset(
    priors,
    sur: str,
    dt: float = 2.0,
    modes: List[Tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)],
    parallel: bool = True
):
    """
    Generate waveform dataset across a parameter grid and save directly to an HDF5 file.
    """
    qs_grid = np.linspace(1.000, 8.0, 50)
    chi1_grid = np.linspace(-0.8, 0.8, 50)
    chi2_grid = np.linspace(-0.8, 0.8, 50)
    params = np.stack(np.meshgrid(qs_grid, chi1_grid, chi2_grid), axis=-1).reshape(-1, 3)
    chunksize = len(params[:, 0]) // 4
    N = len(params[:, 0])

    output_path = f'./data/{sur}_hm_dataset_dt{dt}.hdf'
    with h5py.File(output_path, 'w', libver='latest', swmr=True) as file:
        file.swmr_mode = True
        file.create_dataset(name='Waveforms', shape=(N, len(modes), int(4096 / dt)), dtype=np.complex128)
        dset_snrs = file.create_dataset(name='Parameters', shape=(N, params.shape[-1]), dtype=np.float32)
        file['Parameters'].attrs['names'] = list(['mass_ratio', 'chi_1', 'chi_2'])

        for i in tqdm(range(N // chunksize)):
            params_temp = params[i * chunksize: (i + 1) * chunksize]
            params_dict = {'mass_ratio': params_temp[:, 0], 'chi_1': params_temp[:, 1], 'chi_2': params_temp[:, 2]}
            params_list = convert_dict_to_list_of_dicts(params_dict)
            basedata = gen_hm_data(
                parallel=parallel,
                inj_params=params_list,
                N=chunksize,
                use_tqdm=True,
                whiten=False,
                sur=sur,
                modes=modes,
                dt=dt
            )
            file['Waveforms'][i * chunksize: (i + 1) * chunksize, ...] = basedata.astype(np.complex128)
            file['Parameters'][i * chunksize: (i + 1) * chunksize] = params_temp
