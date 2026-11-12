"""
Injection and recovery with DANSur_HM, using nessai or pocoMC with the likelihood
evaluated in batches (on a GPU with --device cuda).

    pip install "dansur-hm[nessai]"    # or "dansur-hm[pocomc]"
    python examples/batched_injection.py --device cuda --sampler nessai --outdir outdir_nessai

Samples with time, distance and orbital phase marginalised (--marginalize to choose);
they are reconstructed afterwards, in batches. Produces a standard bilby result (and corner plot)
in --outdir.
"""
import argparse

import bilby
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

from dansur.batched_likelihood import BatchedDANSurLikelihood, BatchedNessai, BatchedPocoMC
from dansur.bilby_source import dansur_binary_black_hole

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--sampler", choices=["nessai", "inessai", "pocomc"], default="nessai",
                    help="standard nessai, nessai's importance nested sampler, or pocoMC")
parser.add_argument("--nlive", type=int, default=None, help="default 1000 (nessai), 5000 (inessai)")
parser.add_argument("--n-effective", type=int, default=512, help="pocoMC particles (n_active is half)")
parser.add_argument("--outdir", default="outdir_nessai")
parser.add_argument("--label", default="dansur_injection")
parser.add_argument("--seed", type=int, default=150914)
parser.add_argument("--threads", type=int, default=4, help="pytorch threads for the sampler's flows")
parser.add_argument("--marginalize", nargs="*", choices=["time", "distance", "phase"],
                    default=["time", "distance", "phase"], help="parameters to marginalise over (default all)")
args = parser.parse_args()

bilby.core.utils.random.seed(args.seed)
duration, sampling_frequency, minimum_frequency = 4.0, 2048.0, 20.0
injection = dict(chirp_mass=28.0, mass_ratio=0.5, chi_1=0.3, chi_2=-0.2, luminosity_distance=800.0,
                 theta_jn=0.8, phase=1.3, ra=1.2, dec=-0.4, psi=0.6, geocent_time=1126259642.4)
start_time = injection["geocent_time"] + 2 - duration

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
    time_domain_source_model=dansur_binary_black_hole,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
)
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection, raise_error=False)
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

# Aligned spins within DANSur's range (|chi| <= 0.8), q = m1/m2 <= 5
priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(22, 35)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.2, 1)
for key in ["chi_1", "chi_2"]:
    priors[key] = bilby.core.prior.Uniform(-0.8, 0.8, name=key)
priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(100, 3000, name="luminosity_distance")
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection["geocent_time"] - 0.1, injection["geocent_time"] + 0.1, name="geocent_time")

# bilby's likelihood sets up the time/distance marginalisation (and fixes those parameters
# in `priors`); the batched likelihood follows it, adding the orbital phase. bilby sums over
# arrival times every 2 / sampling_frequency (~1 ms), too coarse for a loud signal; the
# batched likelihood refines that grid by the SNR, so bilby's time jitter isn't needed.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifos, waveform_generator, priors=priors, time_marginalization="time" in args.marginalize,
    distance_marginalization="distance" in args.marginalize, jitter_time=False)
batched = BatchedDANSurLikelihood(likelihood, device=args.device, time_upsampling="auto",
                                  orbital_phase_marginalization="phase" in args.marginalize)
gpu = args.device.startswith("cuda")

kwargs = dict(outdir=args.outdir, label=args.label, injection_parameters=injection, use_ratio=True,
              pytorch_threads=args.threads)
if args.sampler == "pocomc":  # its flow trains on the CPU
    sampler = BatchedPocoMC(batched, likelihood, priors, random_state=args.seed, n_effective=args.n_effective,
                            n_active=args.n_effective // 2, n_total=5000, **kwargs)
else:
    kwargs["seed"] = args.seed
    if gpu:  # train and sample the flows on the GPU too
        kwargs["training_config"] = dict(device_tag=args.device)
    if args.sampler == "inessai":  # importance nested sampler, its own defaults otherwise
        kwargs["importance_nested_sampler"] = True
        if args.nlive:
            kwargs["nlive"] = args.nlive
    else:
        kwargs.update(nlive=args.nlive or 1000, flow_class="gwflowproposal")
    sampler = BatchedNessai(batched, likelihood, priors, **kwargs)
result = bilby.run_sampler(
    likelihood, priors, sampler=sampler, outdir=args.outdir, label=args.label,
    injection_parameters=injection, conversion_function=batched.generate_all_bbh_parameters,
)
result.plot_corner(parameters=["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
                               "theta_jn", "ra", "dec", "geocent_time"])
print(f"log Bayes factor {result.log_bayes_factor:.2f}, {len(result.posterior)} posterior samples")
for key in ["chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"]:
    lo, med, hi = np.percentile(result.posterior[key], [5, 50, 95])
    print(f"{key:20s} {med:9.3f}  [{lo:9.3f}, {hi:9.3f}]   injected {result.injection_parameters[key]:9.3f}")
