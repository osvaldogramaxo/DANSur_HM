# gwsurrogate must be imported before torch: the other order shifts CPU outputs by
# ~5e-5 relative (even with MKL_CBWR pinned). This order reproduces the published results.
import gwsurrogate  # noqa: F401
import torch
import os
import warnings
import cmath
import gwtools as _gwtools 
import numpy as np
import numba
from gwtools import gwutils as _gwutils
from gwsurrogate import SurrogateEvaluator
from typing import Any
# from gwsurrogate.surrogate import EvaluateSingleModeSurrogate
# from EvaluateSingleModeSurrogate import adjust_merger_phase

@numba.njit
def _numba_get_pn_amplitude(l, m, v, q, chi1z=0.0, chi2z=0.0):
    nu = q / (1.0 + q)**2
    delta = (q - 1.0) / (q + 1.0)
    chi_S = 0.5 * (chi1z + chi2z)
    chi_A = 0.5 * (chi1z - chi2z)
    x = v**2
    sqrt_x = v
    H_lm = 0.0 + 0.0j

    if l == 2:
        if m == 2:
            c_0 = 1.0
            c_1 = -107.0/42.0 + 55.0/21.0 * nu
            c_1_5_spin = - (8.0/3.0 * chi_S + 8.0/3.0 * delta * chi_A)
            H_lm = c_0 + c_1 * x + c_1_5_spin * x**1.5
        elif m == 1:
            c_0_5 = (1j / 3.0) * delta
            c_1_5 = (1j / 3.0) * delta * (-17.0/28.0 + 5.0/7.0 * nu)
            c_1_spin = - (1j / 2.0) * (chi_A + delta * chi_S)
            H_lm = c_0_5 * sqrt_x + c_1_spin * x + c_1_5 * x**1.5
    elif l == 3:
        if m == 3:
            c_0_5 = -1j * (3.0 / 4.0) * np.sqrt(15.0 / 14.0) * delta
            c_1_5 = c_0_5 * (-123.0/110.0 + 183.0/44.0 * nu)
            H_lm = c_0_5 * sqrt_x + c_1_5 * x**1.5
        elif m == 2:
            c_1 = (1.0 / 3.0) * np.sqrt(5.0 / 7.0) * (1.0 - 3.0 * nu)
            c_2 = c_1 * (-193.0/90.0 + 145.0/18.0 * nu)
            H_lm = c_1 * x + c_2 * x**2
        elif m == 1:
            c_0_5 = -1j * (1.0 / 12.0) * np.sqrt(1.0 / 14.0) * delta
            c_1_5 = c_0_5 * (-607.0/198.0 + 131.0/33.0 * nu)
            H_lm = c_0_5 * sqrt_x + c_1_5 * x**1.5
    elif l == 4:
        if m == 4:
            c_1 = -(4.0 / 9.0) * np.sqrt(10.0 / 7.0) * (1.0 - 3.0 * nu)
            H_lm = c_1 * x
        elif m == 3:
            # -i: with +i the (4,3) mode is out of phase with NRHybSur3dq8 and SXS
            c_1_5 = -1j * (9.0 / 4.0) * np.sqrt(1.0 / 70.0) * delta * (1.0 - 2.0 * nu)
            H_lm = c_1_5 * x**1.5
        elif m == 2:
            c_2 = (1.0 / 63.0) * np.sqrt(2.0) * (1.0 - 7.0 * nu + 13.0 * nu**2)
            H_lm = c_2 * x**2
        elif m == 1:
            c_1_5 = 1j * (1.0 / 84.0) * np.sqrt(1.0 / 10.0) * delta * (1.0 - 2.0 * nu)
            H_lm = c_1_5 * x**1.5
    elif l == 5:
        if m == 5:
            c_1_5 = 1j * (625.0 / 384.0) * np.sqrt(22.0 / 21.0) * delta * (1.0 - 2.0 * nu)
            H_lm = c_1_5 * x**1.5

    return H_lm

@numba.njit
def _numba_get_pn_amplitude_batch(l, m, v, q, chi1z, chi2z):
    n = len(v)
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = _numba_get_pn_amplitude(l, m, v[i], q[i], chi1z[i], chi2z[i])
    return out

class DANSur(SurrogateEvaluator):


    def __init__(self, modes_list = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)], device='cpu', path=None):
        """
        gwsurrogate-style wrapper for DANSur_HM. Intrinsic parameters are mass ratio (q) and aligned spins (chiA0, chiB0).
        Single waveforms: q, chiA0, chiB0 are floats (or spins as [0, 0, chiz]).
        Batches: q of shape (N,) and spins of shape (N, 3) with zero in-plane components; outputs then have shape (N, n_times).
        device: torch device for the network and all post-processing (e.g. 'cuda'); results are returned as numpy arrays.
        """
        self.name = 'DANSur'
        self.device = device

        # load the dimensionless surrogate
        self._sur_dimless = self._load_dimless_surrogate(path)

        self._domain_type = 'Time'
        if self._domain_type not in ['Time', 'Frequency']:
            raise Exception('Invalid domain_type.')

        # Get some useful keywords, set missing keywords to default values
        self.keywords = {}
        self._check_keywords_and_set_defaults()

        self.soft_param_lims = [8,0.8]
        self.hard_param_lims = [10,0.8]

        

        
        self.modes_list = modes_list
        
        print('Loaded %s model'%self.name + ' on %s'%self.device)
          #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def amp_phase(self,h):
        """Get amplitude and phase of waveform, h = A*exp(i*phi)"""
        return np.abs(h), np.unwrap(np.angle(h))
    
    def phi_merger(self,h):
        """Phase of mode at amplitude's discrete peak. h = A*exp(i*phi)."""

        amp = np.abs(h)
        phase = np.unwrap(np.angle(h))
        argmax_amp = np.argmax(amp)

        return phase[argmax_amp] 
    def get_phi_init(self,h):
        """Phase of mode at time t=0. h = A*exp(i*phi)."""

        amp = np.abs(h)
        phase = np.unwrap(np.angle(h))
        return phase[0]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def adjust_merger_phase(self,h,phiref):
        """Modify GW mode's phase such that at time of amplitude peak, t_peak, we have phase(t_peak) = phiref"""

        phimerger = self.phi_merger(h)
        phiadj    = phiref - phimerger

        return h * np.exp(1j * phiadj)
    def adjust_init_phase(self,h,phi_init):
        return h * np.exp(1j * phi_init)
    def compute_initial_velocity(self, t_grid, phi_22_net):
        """
        Computes the initial characteristic velocity v0 from the (2,2) phase.
        t_grid: Array of dimensionless time values (e.g., -3996 to 1000).
        phi_22_net: Array of the (2,2) phase predicted by the network.
        """
        # Use a finite difference over the first few time steps to get the initial frequency
        dt = t_grid[1] - t_grid[0]
        
        # omega_22 = d(phi_22)/dt
        omega_22_initial = (phi_22_net[1] - phi_22_net[0]) / dt
        
        # The orbital frequency is half the (2,2) frequency
        omega_orb_initial = omega_22_initial / 2.0
        
        # Characteristic velocity v = omega_orb^(1/3)
        v0 = omega_orb_initial ** (1.0 / 3.0)
        return v0

    def get_pn_phase_offset(self, l, m, v0, q, chi1z, chi2z):
        """
        Extracts the initial phase offset xi_lm from the PN amplitude.
        """
        if l == 2 and m == 2:
            return 0.0  # The (2,2) mode is the gauge reference
            
        # ------------------------------------------------------------------
        # NOTE: Replace this block with a call to your preferred PN library.
        # For example, using lalsimulation.SimInspiralPNModeAmplitudes
        # Evaluate the complex amplitude H_lm^{PN} at velocity v0.
        # ------------------------------------------------------------------
        H_lm_complex = self.get_pn_amplitude(l, m, v0, q, chi1z, chi2z)
        
        # The offset xi_lm is the complex argument of the PN amplitude
        xi_lm = np.angle(H_lm_complex)
        return xi_lm

    def spin_weighted_spherical_harmonic(self, spin, l, m, inclination, phi_ref):
        """
        Evaluates the spin-weighted spherical harmonic.
        Replace with `spherical_functions` or `quaternionic` module calls.
        """
        # Placeholder return
        return 1.0 + 0.0j

    def recombine_waveform(self, t_grid, network_predictions, q, chi1z, chi2z, inclination, phi_ref):
        """
        Recombines the neural network predictions into the full observer strain.
        
        network_predictions: Dictionary keyed by (l, m) tuples containing
                            (A_net, phi_net) arrays predicted by the networks.
        """
        # 1. Determine the initial velocity from the (2,2) phase
        phi_22_net = network_predictions[(2, 2)][1]
        v0 = self.compute_initial_velocity(t_grid, phi_22_net) 
        
        total_strain = np.zeros_like(t_grid, dtype=complex)
        
        for (l, m), (A_net, phi_net) in network_predictions.items():
            if m < 0:
                continue # We handle m < 0 via symmetry
                
            # 2. Compute the PN phase offset
            xi_lm = self.get_pn_phase_offset(l, m, v0, q, chi1z, chi2z)
            
            # 3. Shift the predicted phase
            phi_phys = phi_net + xi_lm
            
            # 4. Construct the complex strain for the +m mode
            h_lm = A_net * np.exp(-1j * phi_phys)
            
            # 5. Project onto the spin-weighted spherical harmonic
            Y_lm = self.spin_weighted_spherical_harmonic(-2, l, m, inclination, phi_ref)
            total_strain += h_lm * Y_lm
            
            # 6. Apply equatorial symmetry for the -m mode
            if m > 0:
                h_l_minus_m = ((-1)**l) * np.conjugate(h_lm)
                Y_l_minus_m = self.spin_weighted_spherical_harmonic(-2, l, -m, inclination, phi_ref)
                total_strain += h_l_minus_m * Y_l_minus_m
                
        return total_strain
    def get_pn_amplitude(self, l: int, m: int, v: Any, q: Any, chi1z: Any = 0.0, chi2z: Any = 0.0) -> Any:
        if isinstance(v, np.ndarray):
            n = len(v)
            q_arr = q if isinstance(q, np.ndarray) else np.full(n, q)
            chi1_arr = chi1z if isinstance(chi1z, np.ndarray) else np.full(n, chi1z)
            chi2_arr = chi2z if isinstance(chi2z, np.ndarray) else np.full(n, chi2z)
            return _numba_get_pn_amplitude_batch(l, m, v, q_arr, chi1_arr, chi2_arr)
        else:
            return _numba_get_pn_amplitude(l, m, v, q, chi1z, chi2z)

    def compute_initial_phase_offsets(self, modes_list, v0, q, chi1z=0.0, chi2z=0.0):
        """
        Utility function to compute the exact phase shift xi_lm for a list of modes.
        
        Parameters:
        -----------
        modes_list : list of tuples, e.g., [(2,2), (2,1), (3,3), (4,4)]
        
        Returns:
        --------
        phase_offsets : dict, Mode tuples mapped to their float phase offset in radians.
        """
        phase_offsets = {}
        for (l, m) in modes_list:
            if l == 2 and m == 2:
                phase_offsets[(l, m)] = 0.0
                continue
                
            H_lm = self.get_pn_amplitude(l, m, v0, q, chi1z, chi2z)
            
            if H_lm == 0.0j:
                # Fallback if a mode is not yet implemented
                phase_offsets[(l, m)] = 0.0
            else:
                phase_offsets[(l, m)] = cmath.phase(H_lm)
                
        return phase_offsets

    def _evaluate_chunk(self, x, q, t_scale, amp_scale, inclination, phi_ref,
                        times, taper_end_duration):
        """Network evaluation and post-processing of one batch chunk, in torch on self.device.

        Follows the dtypes of the original numpy implementation: complex64 network
        output, float64/complex128 from the phase alignment on.
        Returns (domain, h) as torch tensors on self.device; domain is None if
        `times` is given, h is the strain if `inclination` is given, else a dict of modes.
        """
        dev = self.device
        n = x.shape[0]
        h = self._sur_dimless(x.to(dev))  # (n, n_modes, L), complex64
        L = h.shape[-1]
        modes = [tuple(mode) for mode in self.modes_list]

        # Time grid in M, shifted so the peak of sum_lm |h_lm|^2 is at t = 0
        base = torch.arange(-4096 + 100, 100, 4096 / L, dtype=torch.float64, device=dev)
        l2_norm_sq = torch.zeros(n, L, dtype=torch.float64, device=dev)
        for k in range(len(modes)):
            l2_norm_sq += (h[:, k].abs() ** 2).double()
        peak_idx = l2_norm_sq.argmax(dim=-1)
        domain = base - base[peak_idx][:, None]

        # Phase of each mode relative to (2,2) from PN amplitudes at the initial velocity
        angle_22 = torch.angle(h[:, modes.index((2, 2)), :2])
        omega_22 = (angle_22[:, 1] - angle_22[:, 0]).double() / (domain[:, 1] - domain[:, 0])
        v0 = ((omega_22 / 2.0) ** (1.0 / 3.0)).cpu().numpy()
        h_modes = {}
        for k, (l, m) in enumerate(modes):
            hlm = h[:, k].conj().to(torch.complex128)
            if (l, m) != (2, 2):
                H_lm = self.get_pn_amplitude(l, m, v0, q.numpy(), x[:, 1].numpy(), x[:, 2].numpy())
                hlm = hlm * torch.exp(1j * torch.as_tensor(np.angle(H_lm), device=dev))[:, None]
            h_modes[(l, m)] = hlm

        # Planck taper over the last taper_end_duration M
        if taper_end_duration is not None:
            tol = 0.005
            val = (domain[:, -1:] - domain) / taper_end_duration
            v = val.clamp(tol, 1.0 - tol)
            sigma = 1.0 / (torch.exp(1.0 / v - 1.0 / (1.0 - v)) + 1.0)
            sigma = torch.where(val < tol, 0.0, torch.where(val > 1.0 - tol, 1.0, sigma))
            h_modes = {mode: hlm * sigma for mode, hlm in h_modes.items()}

        if inclination is not None:
            # Strain h = h+ - i hx at (inclination, pi/2 - phi_ref), LAL convention;
            # for nonprecessing systems the m<0 modes follow from equatorial symmetry
            from gwsurrogate.surrogate import _sYlm
            phi = np.pi / 2 - phi_ref
            out = torch.zeros(n, L, dtype=torch.complex128, device=dev)
            for (l, m), hlm in h_modes.items():
                out += _sYlm(-2, l, m, inclination, phi) * hlm
                if m > 0 and not self.keywords['Precessing']:
                    out += (-1) ** l * _sYlm(-2, l, -m, inclination, phi) * hlm.conj()
        else:
            out = h_modes
            if not self.keywords['Precessing']:
                for (l, m) in list(h_modes):
                    if m > 0 and (l, -m) not in h_modes:
                        h_modes[(l, -m)] = (-1) ** l * h_modes[(l, m)].conj()

        # Physical units
        domain = domain * t_scale.to(dev)[:, None]
        if amp_scale is not None:
            a = amp_scale.to(dev)[:, None]
            out = out * a if inclination is not None else {k: v * a for k, v in out.items()}

        if times is None:
            return domain, out

        # Linear interpolation onto `times` (as np.interp), zero outside each waveform
        t = torch.as_tensor(times, dtype=torch.float64, device=dev).expand(n, -1).contiguous()
        j_raw = torch.searchsorted(domain.contiguous(), t, right=True) - 1
        j = j_raw.clamp(0, L - 2)
        x0, x1 = domain.gather(1, j), domain.gather(1, j + 1)
        inside = (t >= domain[:, :1]) & (t <= domain[:, -1:])
        jj = j[..., None].expand(-1, -1, 2)

        def interp(f):
            f = torch.view_as_real(f)
            f0, f1 = f.gather(1, jj), f.gather(1, jj + 1)
            res = (f1 - f0) / (x1 - x0)[..., None] * (t - x0)[..., None] + f0
            res = torch.where((j_raw == L - 1)[..., None], f[:, -1:], res)
            res = torch.where(inside[..., None], res, 0.0)
            return torch.view_as_complex(res.contiguous())

        out = interp(out) if inclination is not None else {k: interp(v) for k, v in out.items()}
        return None, out

    def _load_dimless_surrogate(self, path=None):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to this._sur_dimless() in the __call__ function of this class.
        See NRHybSur3dq8 for an example.
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'DANSur_HM_jitted.pt')
        DANSurModel = torch.jit.load(path, map_location=self.device)
        # The exported invPCA layers store the export device as a string and move
        # their inputs and bases there on every call; map_location doesn't update it.
        for module in DANSurModel.modules():
            if hasattr(module, 'device'):
                module.device = str(self.device)
        return DANSurModel
    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example:
            For NRHybSur3dq8: x = [q, chiA0, chiB0].
            For NRSur7dq4: x = [q, chiA, chiB], where chiA/chiB are vectors of
                size 3.
        """
        # print(q, chiA0, chiB0)
        # print(q.shape, chiA0.shape, chiB0.shape)
        # check inputs are torch tensors. If not, convert them
        #if not isinstance(q, torch.Tensor):
        #    q = torch.tensor(q)
        #if not isinstance(chiA0, torch.Tensor):
        #    chiA0 = torch.tensor(chiA0)
        #if not isinstance(chiB0, torch.Tensor):
        #    chiB0 = torch.tensor(chiB0)
        # print((chiA0).shape)
        # print(np.sum(chiA0, axis=-1))
        if (chiA0).shape[-1] != 1:
            if (chiA0.sum(dim=-1) != chiA0[...,-1]).any():
                raise Exception('chiA0 should have only the z component if 3 dimensional')
            chiA0 = chiA0[...,2]
        if (chiB0).shape[-1] != 1:
            if (chiB0.sum(dim=-1) != chiB0[...,-1]).any():
                raise Exception('chiB0 should have only the z component if 3 dimensional')
            chiB0 = chiB0[...,2]


        if len(q.shape) == 0:
                q = q.unsqueeze(0)
        if len(chiA0.shape) == 0:
                chiA0 = chiA0.unsqueeze(0)
        if len(chiB0.shape) == 0:
                chiB0 = chiB0.unsqueeze(0)
        if (q<=1).all(): q = 1./q

        # check for batch dimension. If not present, add it
        
        # if len(q.shape) == 1:
        #     q = q.unsqueeze(-1)
        #if (chiA0).shape[-1] != 1:
        #    chiA0 = chiA0[...,2]
        #if (chiB0).shape[-1] != 1:
        #    chiB0 = chiB0[...,2]
        # print(q, chiA0, chiB0)
        # print(chiA0.shape,chiB0.shape)

        q_in = q if (q<=1.).all() else 1./q
        torch_inputs = torch.stack((q_in, chiA0, chiB0)).float().T
        if len(torch_inputs.shape) == 3:
            torch_inputs = torch_inputs.squeeze(0)
        return torch_inputs
    def _check_params(self, q, chiA0, chiB0, precessing_opts, tidal_opts,
            par_dict):
        """ Checks that the parameters are valid.

            Raises a warning if outside self.soft_param_lims and
            raises an error if outside self.hard_param_lims.
            If these are None, skips the checks.

            Also some sanity checks for precessing and tidal models.
        """
        ## Allow violations within this value.
        # Sometimes, chi can be 1+1e-16 due to machine precision limitations,
        # this will ignore such cases
        grace = 1e-14

        chiAmag = torch.linalg.norm(chiA0, dim=-1) if len(chiA0.shape)!=1 else chiA0
        chiBmag = torch.linalg.norm(chiB0, dim=-1) if len(chiB0.shape)!=1 else chiB0

        if not self.keywords['Precessing']:
            # print(chiA0.shape,chiB0.shape)
            # print(chiA0[..., :2])
            # print(torch.linalg.norm(chiA0[..., :2]) )
            if chiA0.shape[-1]==1:
                assert chiB0.shape[-1]==1

            else:
                if (torch.linalg.norm(chiA0[..., :2]) > grace
                        or torch.linalg.norm(chiB0[..., :2]) > grace):
                    raise Exception('Got precessing spins for a nonprecessing '
                        'model')

                if precessing_opts is not None:
                    raise Exception('precessing_opts should be None for '
                            'nonprecessing models')


        if self.keywords['Tidal']:
            if (tidal_opts is None) or (('Lambda1' not in tidal_opts.keys())
                    or ('Lambda2' not in tidal_opts.keys())):
                raise Exception('Tidal parameters Lambda1 and Lambda2 should '
                        'be passed through tidal_opts for this model.')
        else:
            if tidal_opts is not None:
                raise Exception('tidal_opts should be None for nontidal '
                        'models')


        # Extrapolation checks
        if self.hard_param_lims is not None:
            qMax = self.hard_param_lims[0]
            chiMax = self.hard_param_lims[1]
            if (q > qMax + grace).any() or (q < 0.99).any():
                # print(q.shape)
                raise Exception(f'Mass ratio q={q} is outside allowed '
                    f'range: 1<=q<={qMax}')
            # print(chiAmag, chiAmag.shape)
            if (chiAmag > chiMax + grace).any():
                raise Exception(f'Spin magnitude of BhA={chiAmag} is outside '
                    f'allowed range: chi<={chiMax}')
            if (chiBmag > chiMax + grace).any():
                raise Exception(f'Spin magnitude of BhB={chiBmag} is outside '
                    f'allowed range: chi<={chiMax}')

        if self.soft_param_lims is not None:
            qMax = self.soft_param_lims[0]
            chiMax = self.soft_param_lims[1]
            if (q > qMax).any():
                
                warnings.warn('Mass ratio q=%.4f is outside training '
                    'range: 1<=q<=%.4f'%(q.max(), qMax))
            if (chiAmag > chiMax).any():
                warnings.warn('Spin magnitude of BhA=%.4f is outside '
                    'training range: chi<=%.4f'%(chiAmag, chiMax))
            if (chiBmag > chiMax).any():
                warnings.warn('Spin magnitude of BhB=%.4f is outside '
                    'training range: chi<=%.4f'%(chiBmag, chiMax))
    # @torch.compile(mode='reduce-overhead', fullgraph=True, dynamic=True)
    @staticmethod
    def to_wave(x):
        wav = x[...,:x.shape[-1]//2]*torch.exp(1j*x[...,x.shape[-1]//2:])
        # wav = torch.roll(wav, 50, dims=-1)
        return wav
    @torch.inference_mode()
    def __call__(self, q, chiA0, chiB0, M=None, dist_mpc=None, f_low=0,
        f_ref=None, dt=None, df=None, times=None, freqs=None,
        mode_list=None, ellMax=None, inclination=None, phi_ref=0,
        precessing_opts=None, tidal_opts=None, par_dict=None,
        units='dimensionless', skip_param_checks=False,
        taper_end_duration=None, DEBUG=False):
        r"""
    INPUT
    =====
    q :         Mass ratio, mA/mB >= 1.
    chiA0:      Dimensionless spin vector of the heavier black hole at
                reference epoch.
    chiB0:      Dimensionless spin vector of the lighter black hole at
                reference epoch.

                This follows the same convention as LAL, where the spin
                components are defined as:
                \chi_z = \chi \cdot \hat{L}, where L is the orbital angular
                    momentum vector at the epoch.
                \chi_x = \chi \cdot \hat{n}, where n = body2 -> body1 is the
                    separation vector at the epoch. body1 is the heavier body.
                \chi_y = \chi \cdot \hat{L \cross n}.
                These spin components are frame-independent as they are
                defined using vector inner products. This is equivalent to
                specifying the spins in the coorbital frame used in the
                surrogate papers.

    M, dist_mpc: Either specify both M and dist_mpc or neither.
        M        :  Total mass (solar masses). Default: None.
        dist_mpc :  Distance to binary system (MegaParsecs). Default: None.

    f_low :     Instantaneous initial frequency of the (2, 2) mode. In
                practice, this is estimated to be twice the initial orbital
                frequency in the coprecessing frame. Note: the coprecessing
                frame is the minimal rotation frame of arXiv:1110.2965.

                f_low should be in cycles/M if units = 'dimensionless',
                should be in Hertz if units = 'mks'.
                If 0, the entire waveform is returned.
                Default: None, must be specified by user.

                NOTE: For some models like NRSur7dq4, f_low=0 is recommended.
                The role of f_low is only to truncate the lower frequencies
                before returning the waveform. Since this model is already
                very short, this truncation is not required. On the other hand,
                f_ref is used to set the reference epoch, and can be freely
                specified.

                WARNING: Using f_low=0 with a small dt (like 0.1M) can lead to
                very expensive evaluation for hybridized surrogates like
                NRHybSur3dq8.

    f_ref:      Frequency used to set the reference epoch at which the
                reference frame is defined and the spins are specified.
                See below for definition of the reference frame.
                Should be in cycles/M if units = 'dimensionless', should be
                in Hertz if units = 'mks'.
                Default: If f_ref is not given, we set f_ref = f_low. If
                f_low is 0, this corresponds to the initial index.

                For time domain models, f_ref is used to determine a t_ref,
                such that the orbital frequency in the coprecessing frame
                equals f_ref/2 at t=t_ref.

    dt, df :    Time/Frequency step size, specify at most one of dt/df,
                depending on whether the surrogate is a time/frequency domain
                surrogate.
                Default: None. If None, the internal domain of the surrogate is
                used, which can be nonuniformly sampled.
                dt (df) Should be in M (cycles/M) if units = 'dimensionless',
                should be in seconds (Hertz) if units = 'mks'. Do not specify
                times/freqs if using dt/df.


    times, freqs:
                Array of time/frequency samples at which to evaluate the
                waveform, depending on whether the surrogate is a
                time/frequency domain surrogate. time (freqs) should be in
                M (cycles/M) if units = 'dimensionless', should be in
                seconds (Hertz) if units = 'mks'. Do not specify dt/df if
                using times/freqs. Default None.

    ellMax:     Maximum ell index for modes to include. All available m
                indicies for each ell will be included automatically. The 
                m<0 modes will automatically be included for nonprecessing
                models. 
                Default: None, in which case all available ells will be
                included. 

    mode_list : A list of (ell, m) modes tuples to be included. Valid only
                for nonprecessing models.

                Example: mode_list = [(2,2),(2,1)].
                Default: None, in which case all available modes are included.

                At most one of ellMax and mode_list can be specified.

                Note: mode_list is allowed only for nonprecessing models; for
                precessing models use ellMax. For precessing systems, all m
                indices of a given ell index mix with each other, so there is
                no clear hierarchy. To get the individual modes just don't
                specify inclination and a dictionary of modes will be returned.

                Note: When the inclination is set, the m<0 modes are 
                automatically included. For example, passing mode_list = [(2,2)] 
                will include the (2,2) and (2,-2) modes in the computation of
                the strain.

                Note: When the inclination is None, the m<0 modes are
                automatically generated.

    inclination : Inclination angle between the orbital angular momentum
                direction at the reference epoch and the line-of-sight to the
                observer. If inclination is None, the mode data is returned
                as a dictionary.
                Default: None.

    phi_ref :   The azimuthal angle on the sky of the source frame following
                the LAL convention.
                Default: 0.

                If inclination/phi_ref are specified, the complex strain (h =
                hplus -i hcross) evaluated at (inclination, pi/2 - phi_ref) on
                the sky of the reference frame is returned. This follows the
                same convention as LAL. See below for definition of the
                reference frame.

    precessing_opts:
                A dictionary containing optional parameters for a precessing
                surrogate model. Default: None.
                Allowed keys are:
                init_orbphase: The orbital phase in the coprecessing frame
                    at the reference epoch.
                    Default: 0, in which case the coorbital frame and
                    coprecessing frame are the same.
                init_quat: The unit quaternion (length 4 vector) giving the
                    rotation from the coprecessing frame to the inertial frame
                    at the reference epoch.
                    Default: None, in which case the coprecessing frame is the
                    same as the inertial frame.
                return_dynamics:
                    Return the frame dynamics and spin evolution along with
                    the waveform. Default: False.
                Example: precessing_opts = {
                                    'init_orbphase': 0,
                                    'init_quat': [1,0,0,0],
                                    'return_dynamics': True
                                    }

    tidal_opts:
                A dictionary containing optional parameters for a tidal
                surrogate model. Default: None.
                Allowed keys are:
                Lambda1: The tidal deformability parameter for the heavier
                    object.
                Lambda2: The tidal deformability parameter for the lighter
                    object.
                Example: tidal_opts = {'Lambda1': 200, 'Lambda2': 300}


    par_dict:   A dictionary containing any additional parameters needed for a
                particular surrogate model. Default: None.

    units:      'dimensionless' or 'mks'. Default: 'dimensionless'.
                If 'dimensionless': Any of f_low, f_ref, dt, df, times and
                    freqs, if specified, must be in dimensionless units. That
                    is, dt/times should be in units of M, while f_ref, f_low
                    and df/freqs should be in units of cycles/M.
                    M and dist_mpc must be None. The waveform and domain are
                    returned as dimensionless quantities as well.
                If 'mks': Any of f_low, f_ref, dt, df, times and freqs, if
                    specified, must be in MKS units. That is, dt/times should
                    be in seconds, while f_ref, f_low and df/freqs should be
                    in Hz. M and dist_mpc must be specified. The waveform and
                    domain are returned in MKS units as well.


    skip_param_checks :
                Skip sanity checks for inputs. Use this if you want to
                extrapolate outside allowed range. Default: False.

    taper_end_duration:
                Taper the last TAPER_END_DURATION (M) of a time-domain waveform
                in units of M. For exmple, passing 40 will taper the last 40M.
                When set to None, no taper is applied
                Default: None.

    RETURNS
    =====

    domain, h, dynamics


    domain :    Array of time/frequency samples corresponding to h and
                dynamics, depending on whether the surrogate is a
                time/frequency domain model. This is the same as times/freqs
                if times/freqs are given as an inputs.
                For time domain models the time is set to 0 at the peak of
                the waveform. The time (frequency) values are in M (cycles/M)
                if units = 'dimensionless', they are in seconds (Hertz) if
                units = 'mks'

    h :         The waveform.
                    If inclination is specified, the complex strain (h = hplus
                    -i hcross) evaluated at (inclination, pi/2 - phi_ref) on
                    the sky of the reference frame is returned. This follows
                    the LAL convention, see below for details.  This includes
                    all modes given in the ellMax/mode_list argument. For
                    nonprecessing systems the m<0 modes are automatically
                    deduced from the m>0 modes. To see if a model is precessing
                    check self.keywords.

                    Else, h is a dictionary of available modes with (l, m)
                    tuples as keys. For example, h22 = h[(2,2)].

                    If M and dist_mpc are given, the physical waveform
                    at that distance is returned. Else, it is returned in
                    code units: r*h/M extrapolated to future null-infinity.

    dynamics:   A dict containing the frame dynamics and spin evolution. This
                is None for nonprecessing models. This is also None if
                return_dynamics in precessing_opts is False (Default).

                The dynamics include (L=len(domain)):

                q_copr = dynamics['q_copr']
                    The quaternion representing the coprecessing frame with
                    shape (4, L)
                orbphase = dynamics['orbphase']
                    The orbital phase in the coprecessing frame with length L.
                chiA = dynamics['chiA']
                    The inertial frame chiA with shape (L, 3)
                chiB = dynamics['chiB']
                    The inertial frame chiB with shape (L, 3)


    IMPORTANT NOTES:
    ===============

    The reference frame (or inertial frame) is defined as follows:
        The +ve z-axis is along the orbital angular momentum at the reference
        epoch. The separation vector from the lighter BH to the heavier BH at
        the reference epoch is along the +ve x-axis. The y-axis completes the
        right-handed triad. The reference epoch is set using f_ref.

        Now, if inclination/phi_ref are given, the waveform is evaluated at
        (inclination, pi/2 - phi_ref) in the reference frame. This agrees with
        the LAL convention. See LIGO DCC document T1800226 for the LAL frame
        diagram.
        """

        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q)
            if len(q.shape) == 0:
                q = q.unsqueeze(0)
        if not isinstance(chiA0, torch.Tensor):
            chiA0 = torch.tensor(chiA0)
            if len(chiA0.shape) == 0:
                chiA0 = chiA0.unsqueeze(0)
        if not isinstance(chiB0, torch.Tensor):
            chiB0 = torch.tensor(chiB0)
            if len(chiB0.shape) == 0:
                chiB0 = chiB0.unsqueeze(0)
        if (q<=1).all(): q = 1./q

        # Sanity checks
        if not skip_param_checks:

            if (M is None) ^ (dist_mpc is None):
                raise ValueError("Either specify both M and dist_mpc, or "
                        "neither")

            if (M is not None) ^ (units == 'mks'):
                raise ValueError("M/dist_mpc must be specified if and only if"
                    " units='mks'")

            if (dt is not None) and (self._domain_type != 'Time'):
                raise ValueError("%s is not a Time domain model, cannot "
                        "specify dt"%self.name)

            if (times is not None) and (self._domain_type != 'Time'):
                raise ValueError("%s is not a Time domain model, cannot "
                        "specify times"%self.name)

            if (df is not None) and (self._domain_type != 'Frequency'):
                raise ValueError("%s is not a Frequency domain model, cannot"
                    " specify df"%self.name)

            if (freqs is not None) and (self._domain_type != 'Frequency'):
                raise ValueError("%s is not a Frequency domain model, cannot"
                    " specify freqs"%self.name)

            if (dt is not None) and (times is not None):
                raise ValueError("Cannot specify both dt and times.")

            if (df is not None) and (freqs is not None):
                raise ValueError("Cannot specify both df and freqs.")

            if (f_low is None):
                raise ValueError("f_low must be specified.")

            if (f_ref is not None) and (f_ref < f_low):
                raise ValueError("f_ref cannot be lower than f_low.")

            if (mode_list is not None) and (ellMax is not None):
                raise ValueError("Cannot specify both mode_list and ellMax.")

            if (mode_list is not None) and self.keywords['Precessing']:
                raise ValueError("mode_list is not allowed for precessing "
                        "models, use ellMax instead.")

            if (taper_end_duration is not None) and self._domain_type !='Time':
                raise ValueError("%s is not a Time domain model, cannot taper")

            # more sanity checks including extrapolation checks
            self._check_params(q, chiA0, chiB0, precessing_opts, tidal_opts,
                    par_dict)


        x = self._get_intrinsic_parameters(q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict)
        # print(x)

        # Get scalings from dimensionless units to mks units
        if units == 'dimensionless':
            amp_scale = np.ones_like(q)
            t_scale = np.ones_like(q)
        elif units == 'mks':
            M=np.array(M)
            dist_mpc=np.array(dist_mpc)
            if (not isinstance(M, np.ndarray) or M.ndim == 0) or \
               (not isinstance(dist_mpc, np.ndarray) or dist_mpc.ndim == 0):
                if (isinstance(M, np.ndarray) and M.ndim!=0) or (isinstance(dist_mpc, np.ndarray) and dist_mpc.ndim!=0):
                    if isinstance(M, np.ndarray) and M.ndim != 0:
                        dist_mpc = np.tile(dist_mpc, M.shape[0])
                    elif isinstance(dist_mpc, np.ndarray) and dist_mpc.ndim != 0:
                        M = np.tile(M, dist_mpc.shape[0])
                else:
                    M = np.array([M])
                    dist_mpc = np.array([dist_mpc])
            
            
            amp_scale = \
                M*_gwtools.Msuninsec*_gwtools.c/(1e6*dist_mpc*_gwtools.PC_SI)
            t_scale = _gwtools.Msuninsec * M
        else:
            raise Exception('Invalid units')

        if dt is not None:
            raise ValueError('dt is not supported; pass a uniform `times` array instead.')

        t_scale = torch.as_tensor(np.broadcast_to(t_scale, q.shape), dtype=torch.float64)
        amp_scale = (torch.as_tensor(np.broadcast_to(amp_scale, q.shape), dtype=torch.float64)
                     if np.array(amp_scale != 1).any() else None)

        # Chunked so large batches fit in GPU memory; only final results leave the
        # device, copied straight into preallocated CPU outputs
        out = {}
        for i in range(0, x.shape[0], 1024):
            sl = slice(i, i + 1024)
            domain_c, h_c = self._evaluate_chunk(
                x[sl], q[sl], t_scale[sl], None if amp_scale is None else amp_scale[sl],
                inclination, phi_ref, times, taper_end_duration)
            parts = dict(h_c) if inclination is None else {'h': h_c}
            if domain_c is not None:
                parts['domain'] = domain_c
            for key, t in parts.items():
                if key not in out:
                    out[key] = torch.empty((x.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)
                out[key][sl] = t

        dynamics = None
        domain = out.pop('domain', None)
        interp_h = out['h'].numpy() if inclination is not None else {k: v.numpy() for k, v in out.items()}
        interp_domain = times if times is not None else domain.numpy()

        # Squeeze batch dimension if N = 1
        if q.shape[0] == 1:
            if isinstance(interp_h, dict):
                interp_h = {mode: hlm[0] for mode, hlm in interp_h.items()}
            else:
                interp_h = interp_h[0] # type: ignore
            interp_domain = interp_domain[0] if times is None else interp_domain # type: ignore

        return interp_domain, interp_h, dynamics
