import os
import argparse
import yaml
import torch
from torch import nn
import sys


from dansur.utils.utils import Decoder
from dansur.training.sxs_hm import DANSurEnsemble, load_config, get_device

def main():
    parser = argparse.ArgumentParser(description="Create a jitted version of the trained DANSurEnsemble model")
    parser.add_argument('--config', type=str, default='scripts/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (default: uses models_dir/decoder_mode_summed.pt)')
    parser.add_argument('--output-path', type=str, default=None, help='Output path for jitted model (default: uses models_dir/DANSur_HM_jitted.pt)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")
    
    modes = [tuple(m) for m in config['modes']]
    layers = config['model']['layers']
    decoders = []
    
    # Instantiate the base decoders using the basis from pretraining
    for i, mode in enumerate(modes):
        path = config['model']['pretrain_path_template'].format(mode=mode)
        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        amp_basis = state_dict['amp_basis']
        amp_mean = state_dict['amp_mean']
        phase_basis = state_dict['phase_basis']
        phase_mean = state_dict['phase_mean']
        
        _model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=nn.GELU, device=device)
        _model.float()
        _model.eval()
        decoders.append(_model)
    
    # Create ensemble
    ens = DANSurEnsemble(decoders, modes, device=device)
    ens.float()
    
    # Load trained ensemble weights
    model_path = args.model_path or f"{config['output']['models_dir']}/decoder_mode_summed.pt"
    print(f"Loading trained weights from: {model_path}")
    ens.load_state_dict(torch.load(model_path, map_location=device))
    ens.eval()
    ens(torch.tensor([[0.5, 0.1, 0.1]]))
    plt.plot
    # Create jitted version
    print("Jitting model...")
    jitted_model = torch.jit.script(ens)
    
    # Save jitted model
    output_path = args.output_path or f"{config['output']['models_dir']}/DANSur_HM_jitted.pt"
    jitted_model.save(output_path)
    print(f"Successfully saved jitted model to: {output_path}")

if __name__ == "__main__":
    main()
