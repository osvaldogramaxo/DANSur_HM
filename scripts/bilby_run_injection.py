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
from bilby_run import model_gen_func, _numba_taper_start, _sYlm

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
