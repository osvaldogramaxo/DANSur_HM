# %%
import h5py
import pyseobnr
import numpy as np
import bilby
from bilby.core.prior import Uniform
import sys
og_stderr= sys.stderr
og_stdout = sys.stdout
from ..utils.utils import *
# %%

# %%
bg_ds_pre = []
# noise_path = '/lhome/ext/uv098/uv0982/scratch/ml_mock_data/ds4/background.hdf'
# segment_starts= list(h5py.File(noise_path)['H1'].keys() )


# %%
duration = 4.0 # seconds
sampling_frequency = 1024.0 # Hz

LAL_MTSUN_SI = 4.925490947641266978197229498498379006e-6
mTScaled = 100 * LAL_MTSUN_SI;
f_lower = pow(10.5, -1.5) / (np.pi * mTScaled)
waveform_arguments = dict(
    # waveform_approximant="NRHybSur3dq8",
    waveform_approximant="NRHybSur3dq8",
    minimum_frequency=f_lower,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)



# Create a dictionary of priors for binary black hole parameters
priors = bilby.gw.prior.BBHPriorDict(
    dict(
        # Prior for the mass ratio
        mass_ratio=Uniform(1, 8),
        # Prior for the total mass
        total_mass=1.0,
        # Prior for the spin parameter of the first black hole
        chi_1 = bilby.gw.prior.AlignedSpin(a_prior=Uniform(minimum=0, maximum=0.81, 
                                                    name=None, latex_label=None, unit=None, boundary=None), 
                                                    z_prior=Uniform(minimum=-1, maximum=1, name=None, latex_label=None, unit=None, boundary=None), 
                                                    name='chi_1', latex_label='$\\chi_1$', unit=None, boundary=None, minimum=-0.8, maximum=0.8),
        # # Prior for the spin parameter of the second black hole
        chi_2 =  bilby.gw.prior.AlignedSpin(a_prior=Uniform(minimum=0, maximum=0.81, 
                                                    name=None, latex_label=None, unit=None, boundary=None), 
                                                    z_prior=Uniform(minimum=-1, maximum=1, name=None, latex_label=None, unit=None, boundary=None), 
                                                    name='chi_2', latex_label='$\\chi_2$', unit=None, boundary=None, minimum=-0.8, maximum=0.8),
        
    )
)



# %%
sys.stdout = og_stdout
if __name__ == "__main__":
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Generate and save data to an HDF5 file.')
    parser.add_argument('--sur', type=str, help='Surrogate name', default='NRSur7dq4')
    
    
    # Parse the arguments
    args = parser.parse_args()
    sur_name = args.sur

    # Call the generate_data_ng function with the file_path argument
    generate_dataset(priors, sur_name)
