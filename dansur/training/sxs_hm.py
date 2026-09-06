import sys
import os
# Add workspace root to TEXINPUTS so LaTeX can find local style files
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.environ['TEXINPUTS'] = workspace_root + ':' + os.environ.get('TEXINPUTS', '')

import argparse
import yaml
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_optimizer as topt
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib import rc

# from scipy.stats import binned_statistic_2d


from dansur.training.train_decoder_ensemble import *
from dansur.utils.utils import *
from dansur.utils.sxs_utils import SXSDataset

import seaborn as sns
import matplotlib.colors as colors

sns.set_style('whitegrid')
# Set publication style
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')

# Sizes and style consistency for publication
rc('font', size=14)
rc('axes', labelsize=16)
rc('axes', titlesize=14)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('legend', fontsize=12)
rc('figure', titlesize=14)

#set default plt.figure() #plt.figure(figsize=(8, 6))
# plt.rcParams.update({"figure.figsize": (8, 6)})
#set default plt.tight_layout()
plt.rcParams.update({"figure.autolayout": True})


# Ensure device availability
def get_device():
    if torch.cuda.is_available():
        try:
            torch.tensor([0.0], device='cuda')
        except:
            return 'cpu'
        return 'cuda'
    return 'cpu'

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Model Definitions (from original script) ---

class DANSurEnsemble(nn.Module):
    def __init__(self, decoders, modes_list, device):
        super(DANSurEnsemble, self).__init__()
        self.decoders = nn.ModuleList(decoders).to(device)
        self.modes_list = modes_list
        self.device = device

    def to_wave(self, x):
        len_wav = x.shape[-1]//2
        return x[...,:len_wav] * torch.exp(1j * x[...,len_wav:])

    def forward(self, x):
        outputs = []
        for decoder in self.decoders:
            output = decoder(x)
            outputs.append(output)
        x = torch.stack(outputs, dim=0)
        x = self.to_wave(x)
        x = x.permute(1, 0, 2)
        # if x.shape[-1] == 8192:
        #     x = x[..., ::2]
        return x

# --- Trainer Class ---

class SXSTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Configuration shortcuts
        self.modes = [tuple(m) for m in self.config['modes']]
        self.mode_map = {i: m for i, m in enumerate(self.modes)}
        self.time_limit = self.config.get('time_limit', 172800)
        
        # Raw mode loss config
        raw_cfg = self.config['training'].get('raw_mode_loss', {})
        self.raw_mode_enabled = raw_cfg.get('enabled', False)
        self.raw_mode_weight = raw_cfg.get('weight', 0.5)
        
        # Setup helpers
        self._setup_cover_sphere()
        self.start_time = time.time()
        
    def _setup_cover_sphere(self):
        # Create Cover_Sphere instances for training/validation
        self.cover_forward_sxs = Cover_Sphere(self.mode_map.copy(), n=20, infer_neg_m=True, device=self.device)
        self.cover_forward_nn = Cover_Sphere(self.mode_map.copy(), n=20, infer_neg_m=True, device=self.device)
        
        # Convert to complex float32
        for cf in [self.cover_forward_sxs, self.cover_forward_nn]:
            cf.sylm_values = cf.sylm_values.to(torch.complex64)
            if hasattr(cf, 'neg_m_l_coeffs'):
                cf.neg_m_l_coeffs = cf.neg_m_l_coeffs.to(torch.complex64)

    def train(self, model, optimizer, train_dl, val_dl, scheduler=None):
        print('Training model')
        
        create_graph = isinstance(optimizer, topt.SophiaH)
        bestmodel_weights = model.state_dict()
        bestloss = float('inf')
        worst_mm = float('inf')
        best_ep = 0
        
        train_losses = []
        val_losses = []
        mm_history = []
        
        early_stopping_counter = 0
        es_patience = self.config['training']['early_stopping']['patience']
        es_delta = self.config['training']['early_stopping']['delta']
        num_epochs = self.config['training']['num_epochs']
        verbose = self.config['output']['verbose']
        plotting = self.config['output']['plotting']
        
        if scheduler is not None:
             scheduler = scheduler(optimizer, patience=100, factor=0.5)

        with trange(num_epochs, disable=not verbose) as pbar:
            for epoch in pbar:
                if time.time() - self.start_time > self.time_limit:
                    raise Exception('Time limit reached')

                # -- Training Step --
                model.train()
                epoch_loss = 0
                for params, wf in train_dl:
                    # Raw mode prediction (before sphere projection)
                    raw_pred = model(params.to(self.device))
                    raw_target = wf.to(self.device)
                    
                    # Sphere-projected waveforms
                    wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
                    outputs_wave = self.cover_forward_nn(raw_pred).flatten(0,2)
                    
                    # Sphere-projected mismatch
                    sphere_mm_loss = mymismatch(outputs_wave, wf_wave)
                    sphere_mm_loss = torch.nan_to_num(sphere_mm_loss)
                    
                    power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
                    
                    # Sphere loss (log of mismatch)
                    sphere_loss = torch.log10(sphere_mm_loss.mean())
                    
                    # Raw mode loss (if enabled)
                    if self.raw_mode_enabled:
                        raw_mm_loss = mymismatch(raw_pred.flatten(0,1), raw_target.flatten(0,1))
                        raw_mm_loss = torch.nan_to_num(raw_mm_loss)
                        raw_loss = torch.log10(raw_mm_loss.mean())
                        
                        # Combine losses with weighting
                        combined_mm_loss = (1 - self.raw_mode_weight) * sphere_loss + self.raw_mode_weight * raw_loss
                    else:
                        combined_mm_loss = sphere_loss
                    
                    # Total loss with power diff
                    loss = combined_mm_loss + power_diff.mean()
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(create_graph=create_graph)
                    optimizer.step()
                    epoch_loss = loss.item()

                # -- Validation Step --
                model.eval()
                val_results = self.validate(model, val_dl)
                val_loss = val_results['val_loss']
                mm_loss_valid = val_results['mm_loss_valid']
                
                if torch.isnan(torch.tensor(val_loss)): # Safe check
                    print("Loss is nan")
                    return float('nan'), float('nan')
                
                # Save best model
                improv='False'
                if val_loss < bestloss: # This logic slighty distinct from ES check for saving? Original script saves on ANY improvement vs best?
                    improv='True'
                    # Actually original script checks 'if val_loss.item() < bestloss' again
                    bestmodel_weights = model.state_dict()
                    best_ep = epoch
                    worst_mm = mm_loss_valid.max().item()
                    
                    if plotting:
                        self.plot_diagnostics(epoch, mm_loss_valid, val_results, model)
                # Check early stopping
                if val_loss < bestloss - es_delta:
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= es_patience:
                    print(f"Early stopping at epoch {epoch}. Starting LBFGS fine-tuning...")
                    break
                
                else:
                    # Scheduler step logic from original
                    if scheduler and (scheduler.num_bad_epochs + 1 > scheduler.patience):
                         # Just printing logic handled by scheduler verbose=True usually, but original had print
                         pass
                
                if scheduler:
                    scheduler.step(val_loss)
                
                bestloss = min(val_loss, bestloss)
                
                if verbose:
                    desc = f'Epoch [{epoch+1}/{num_epochs}], Val loss: {val_loss:.2e}, MM: {mm_loss_valid.mean().item():.2e}, Worst: {worst_mm:.2e}, Improvement: {improv}'
                    pbar.set_description(desc)
                
                mm_history.append(mm_loss_valid.mean().cpu().numpy())
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                if epoch % 10 == 0:
                     torch.save(bestmodel_weights, f"{self.config['output']['models_dir']}/decoder_mode_summed.pt")
                     if plotting:
                         self.plot_losses(train_losses, val_losses)

        # -- LBFGS Fine-tuning --
        if self.config['lbfgs']['enabled']:
             bestloss, best_mean_mm = self.fine_tune_lbfgs(model, bestmodel_weights, best_ep, bestloss, train_dl, val_dl)
        else:
             best_mean_mm = np.min(mm_history)

        # Final save
        torch.save(bestmodel_weights, f"{self.config['output']['models_dir']}/decoder_mode_summed.pt")
        # Save jitted model
        jitted_model = torch.jit.script(model)
        jitted_model.save(f"{self.config['output']['models_dir']}/DANSur_HM_jitted.pt")
        return bestloss, best_mean_mm

    def validate(self, model, val_dl):
        with torch.no_grad():
            val_loss = 0.0
            latent_loss_valid = 0.0
            mm_loss_valid = 0.0
            raw_mm_loss_valid = 0.0
            
            for vparams, vwf in val_dl:
                qs = 1/vparams[:,0]
                qs = torch.tile(qs, (1, self.cover_forward_nn.n, self.cover_forward_nn.n)).flatten()
                
                # Raw mode prediction
                raw_pred = model(vparams.to(self.device))
                raw_target = vwf.to(self.device)
                
                # Sphere-projected waveforms
                outputs_wave_valid = self.cover_forward_nn(raw_pred).flatten(0,2)
                vwf_wave = self.cover_forward_sxs(vwf).to(self.device).flatten(0,2)
                
                # Sphere-projected mismatch
                mm_loss_batch = mymismatch(outputs_wave_valid, vwf_wave)
                mm_loss_batch = torch.nan_to_num(mm_loss_batch)
                
                power_diff_valid = nn.L1Loss()(abs(vwf_wave).sum(dim=-1), abs(outputs_wave_valid).sum(dim=-1))
                
                # Sphere loss
                sphere_loss = torch.log10(mm_loss_batch.mean())
                
                # Raw mode loss (if enabled)
                if self.raw_mode_enabled:
                    raw_mm_batch = mymismatch(raw_pred.flatten(0,1), raw_target.flatten(0,1))
                    raw_mm_batch = torch.nan_to_num(raw_mm_batch)
                    raw_loss = torch.log10(raw_mm_batch.mean())
                    raw_mm_loss_valid += raw_mm_batch
                    
                    combined_loss = (1 - self.raw_mode_weight) * sphere_loss + self.raw_mode_weight * raw_loss
                else:
                    combined_loss = sphere_loss
                
                val_loss += combined_loss + power_diff_valid.mean()
                mm_loss_valid += mm_loss_batch

            # Averaging
            val_loss = val_loss / len(val_dl)
            mm_loss_valid = mm_loss_valid / len(val_dl)
            if self.raw_mode_enabled:
                raw_mm_loss_valid = raw_mm_loss_valid / len(val_dl)
            
            return {
                'val_loss': float(val_loss),
                'mm_loss_valid': mm_loss_valid,
                'raw_mm_loss_valid': raw_mm_loss_valid if self.raw_mode_enabled else None,
                'vwf_wave': vwf_wave, # Return last batch for plotting
                'outputs_wave_valid': outputs_wave_valid, # Return last batch for plotting
                'vparams': vparams, # Return last batch params
                'qs': qs # Return last batch qs
            }

    def fine_tune_lbfgs(self, model, best_weights, best_ep, bestloss, train_dl, val_dl):
        print("Starting LBFGS fine-tuning stage...")
        model.load_state_dict(best_weights)
        
        lbfgs_cfg = self.config['lbfgs']
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_cfg['learning_rate'],
            max_iter=lbfgs_cfg['max_iter'],
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn='strong_wolfe'
        )
        
        lbfgs_epochs = lbfgs_cfg['num_epochs']
        lbfgs_bestloss = bestloss
        lbfgs_best_weights = best_weights.copy()
        lbfgs_best_ep = best_ep
        lbfgs_counter = 0
        lbfgs_patience = lbfgs_cfg['early_stopping']['patience']
        lbfgs_delta = lbfgs_cfg['early_stopping']['delta']
        
        lbfgs_train_losses = []
        lbfgs_val_losses = []
        lbfgs_plot_dir = os.path.join(self.config['output']['plots_dir'], 'lbfgs_finetune') if self.config['output']['plotting'] else None
        
        train_iter = iter(train_dl)

        def closure():
            nonlocal train_iter
            lbfgs_optimizer.zero_grad()
            model.train()
            try:
                params, wf = next(train_iter)
            except StopIteration:
                 # Restart iterator
                 train_iter = iter(train_dl)
                 params, wf = next(train_iter)

            # Raw mode prediction
            raw_pred = model(params.to(self.device))
            raw_target = wf.to(self.device)
            
            # Sphere-projected waveforms
            wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
            outputs_wave = self.cover_forward_nn(raw_pred).flatten(0,2)
            
            # Sphere-projected mismatch
            sphere_mm_loss = mymismatch(outputs_wave, wf_wave)
            sphere_mm_loss = torch.nan_to_num(sphere_mm_loss)
            power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
            
            # Sphere loss
            sphere_loss = torch.log10(sphere_mm_loss.mean())
            
            # Raw mode loss (if enabled)
            if self.raw_mode_enabled:
                raw_mm_loss = mymismatch(raw_pred.flatten(0,1), raw_target.flatten(0,1))
                raw_mm_loss = torch.nan_to_num(raw_mm_loss)
                raw_loss = torch.log10(raw_mm_loss.mean())
                
                combined_mm_loss = (1 - self.raw_mode_weight) * sphere_loss + self.raw_mode_weight * raw_loss
            else:
                combined_mm_loss = sphere_loss
            
            loss = combined_mm_loss + power_diff.mean()
            loss.backward()
            return loss

        with trange(lbfgs_epochs, disable=not self.config['output']['verbose']) as pbar:
            for ep in pbar:
                if time.time() - self.start_time > self.time_limit:
                    print("Time limit reached during LBFGS")
                    break
                    
                model.train()
                try:
                    loss = lbfgs_optimizer.step(closure)
                except Exception as e:
                    print(f"LBFGS step failed: {e}")
                    break
                
                # Validation
                model.eval()
                val_res = self.validate(model, val_dl)
                val_loss = val_res['val_loss']
                
                if val_loss < lbfgs_bestloss - lbfgs_delta:
                    lbfgs_bestloss = val_loss
                    lbfgs_counter = 0
                    lbfgs_best_weights = model.state_dict().copy()
                    lbfgs_best_ep = best_ep + ep + 1
                    if self.config['output']['plotting']:
                         self.plot_diagnostics(lbfgs_best_ep, val_res['mm_loss_valid'], val_res, model, suffix='_lbfgs', save_dir=lbfgs_plot_dir)
                elif ep==0 and self.config['output']['plotting']:
                    self.plot_diagnostics(lbfgs_best_ep, val_res['mm_loss_valid'], val_res, model, suffix='_lbfgs', save_dir=lbfgs_plot_dir)
                else:
                    lbfgs_counter += 1
                
                if lbfgs_counter >= lbfgs_patience:
                    print(f"LBFGS early stopping at {ep}")
                    break
                    
                if self.config['output']['verbose']:
                    pbar.set_description(f"LBFGS Ev: {val_loss:.2e}, Best: {lbfgs_bestloss:.2e}")
                
                lbfgs_train_losses.append(loss.item())
                lbfgs_val_losses.append(val_loss)
                
                if self.config['output']['plotting'] and ep % 10 == 0:
                     self.plot_losses(lbfgs_train_losses, lbfgs_val_losses, save_dir=lbfgs_plot_dir, suffix='_lbfgs')

        if self.config['output']['plotting']:
             self.plot_losses(lbfgs_train_losses, lbfgs_val_losses, save_dir=lbfgs_plot_dir, suffix='_lbfgs')
        # Return global best
        if lbfgs_bestloss < bestloss:
             return lbfgs_bestloss, np.min(val_res['mm_loss_valid'].cpu().numpy()) # Approx
        
        return bestloss, 0.0 # Placeholder

    def plot_diagnostics(self, epoch, mm_loss_valid, val_results, model, suffix='', save_dir=None):
        plots_dir = save_dir if save_dir is not None else self.config['output']['plots_dir']
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        vwf_wave = val_results['vwf_wave']
        outputs_wave_valid = val_results['outputs_wave_valid']
        vparams = val_results['vparams']
        qs = val_results['qs']
        
        worst_idx = mm_loss_valid.argmax()
        worst_mm = mm_loss_valid.max().item()

        # Histogram
        n_bins = 1 + int(np.log2(len(mm_loss_valid)).round())
        plt.figure()
        # Handle zeros for log params
        valid_mm = mm_loss_valid[mm_loss_valid > 0].detach().cpu().numpy()
        if len(valid_mm) > 0:
             bins = np.logspace(np.log10(valid_mm.min()), np.log10(valid_mm.max()), n_bins)
             if mm_loss_valid.min() == 0:
                 bins = np.insert(bins, 0, 0.)
        else:
             bins = 10
             
        plt.hist(mm_loss_valid.flatten().detach().cpu().numpy(), bins=bins, histtype='step', color='black')
        plt.xscale('log')
        plt.yscale('log')
        #plt.title(f'Worst MM: {worst_mm:.2e} @ {epoch}')
        # plt.legend()
        plt.xlabel(r'$\mathfrak{M}$')
        plt.ylabel('Count')
        plt.savefig(f'{plots_dir}/sxs_mm_hist{suffix}.png', dpi=300)
        plt.savefig(f'{plots_dir}/sxs_mm_hist{suffix}.pdf', dpi=300)
        plt.close()
        
        # Worst reconstruction
        plt.figure()
        plt.plot(vwf_wave[worst_idx].detach().cpu(), label='Ground truth')
        plt.plot(outputs_wave_valid[worst_idx].detach().cpu(), label='Prediction')
        plt.legend()
        #plt.title(f'Worst MM: {worst_mm:.2e}')
        plt.savefig(f'{plots_dir}/sxs_worst_recon{suffix}.png', dpi=300)
        plt.close()

        # MM vs chi1 vs chi2
        plt.figure()
        from scipy.stats import binned_statistic_2d
        import matplotlib.colors as colors
        bins = 15

        batch_size = len(vparams)
        mm_reshaped = mm_loss_valid.detach().cpu().numpy().reshape(batch_size, -1)
        mean_mm_per_sample = mm_reshaped.mean(axis=1)

        binned_data_2d = binned_statistic_2d(
            vparams[:, 1].detach().cpu().numpy(),
            vparams[:, 2].detach().cpu().numpy(),
            mean_mm_per_sample,
            statistic='mean',
            bins=bins
        )
        x_edges = binned_data_2d.x_edge
        y_edges = binned_data_2d.y_edge

        plt.figure() #plt.figure(figsize=(8, 6))
        plt.imshow(
            binned_data_2d.statistic.T,
            origin='lower',
            aspect='auto',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            norm=colors.LogNorm(),
            cmap='turbo'
        )
        plt.xlabel(r'$\chi_1$')
        plt.ylabel(r'$\chi_2$')
        plt.colorbar(label=r'$\mathfrak{M}$')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/sxs_mm_vs_chi1_chi2{suffix}.png', dpi=300)
        plt.savefig(f'{plots_dir}/sxs_mm_vs_chi1_chi2{suffix}.pdf', dpi=300)
        plt.close()
        
        
        # mm vs q as 2D histogram
        plt.figure() #plt.figure(figsize=(8, 6))
        mms = mm_loss_valid.detach().cpu().numpy()
        qs_np = qs.detach().cpu().numpy()
        
        valid_mask = mms > 0
        mms = mms[valid_mask]
        qs_np = qs_np[valid_mask]
        
        if len(mms) > 0:
            x_bins = np.logspace(np.log10(mms.min()), np.log10(mms.max()), 50)
            y_bins = np.linspace(qs_np.min(), qs_np.max(), 30)
            
            import matplotlib.colors as colors
            plt.hist2d(mms, qs_np, bins=[x_bins, y_bins], norm=colors.LogNorm(), cmap='viridis')
            plt.colorbar(label='Count')
            
        plt.ylabel('q')
        plt.xlabel(r'$\mathfrak{M}$')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/sxs_mm_vs_q{suffix}.png', dpi=300)
        plt.savefig(f'{plots_dir}/sxs_mm_vs_q{suffix}.pdf', dpi=300)
        plt.close()

        # mm average over sphere vs q as 2D histogram
        plt.figure() #plt.figure(figsize=(8, 6))
        batch_size = len(vparams)
        mm_reshaped = mm_loss_valid.detach().cpu().numpy().reshape(batch_size, -1)
        mms_avg = mm_reshaped.mean(axis=1)
        qs_per_sample = 1.0 / vparams[:, 0].detach().cpu().numpy()
        
        valid_mask = mms_avg > 0
        mms_avg = mms_avg[valid_mask]
        qs_per_sample = qs_per_sample[valid_mask]
        
        if len(mms_avg) > 0:
            x_bins = np.logspace(np.log10(mms_avg.min()), np.log10(mms_avg.max()), 25)
            y_bins = np.linspace(qs_per_sample.min(), qs_per_sample.max(), 30)
            
            import matplotlib.colors as colors
            plt.hist2d(mms_avg, qs_per_sample, bins=[x_bins, y_bins], 
            norm=None, cmap='viridis', cmin=1, vmin=1)
            plt.colorbar(label='Count')
            
        plt.ylabel('q')
        plt.xlabel(r'$\mathfrak{M}$')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/sxs_mm_avg_vs_q{suffix}.png', dpi=300)
        plt.savefig(f'{plots_dir}/sxs_mm_avg_vs_q{suffix}.pdf', dpi=300)
        plt.close()

        # 1D binned statistic: mean mismatch vs q
        from scipy.stats import binned_statistic
        plt.figure() #plt.figure(figsize=(8, 6))
        if len(mms_avg) > 0:
            q_bins = np.linspace(qs_per_sample.min(), qs_per_sample.max(), 15)
            bin_means, bin_edges, _ = binned_statistic(qs_per_sample, mms_avg, statistic='mean', bins=q_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            plt.scatter(qs_per_sample, mms_avg, alpha=0.3, color='gray', s=15, label='Samples')
            plt.stairs(bin_means, bin_edges, color='tab:blue', linewidth=2, label=r'Binned Mean $\mathfrak{M}$')
            valid_bins = ~np.isnan(bin_means)
            plt.plot(bin_centers[valid_bins], bin_means[valid_bins], 'o', color='tab:blue', markersize=4)
            plt.legend()
            
        plt.xlabel(r'$q$')
        plt.ylabel(r'$\mathfrak{M}$')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/sxs_mm_avg_binned_vs_q{suffix}.png', dpi=300)
        plt.savefig(f'{plots_dir}/sxs_mm_avg_binned_vs_q{suffix}.pdf', dpi=300)
        plt.close()

        # Mean mismatch vs theta/phi (sky position heatmap)
        n_grid = self.cover_forward_nn.n
        batch_size = len(vparams)
        # mm_loss_valid has shape (batch * n_phi * n_theta,) after flatten(0,2) in validate
        # Cover_Sphere uses meshgrid(theta, phi) which produces (n_phi, n_theta) shaped arrays
        # So after reshape: (batch, n_phi, n_theta)
        try:
            mm_reshaped = mm_loss_valid.detach().cpu().numpy().reshape(batch_size, n_grid, n_grid)
            mean_mm_sky = np.mean(mm_reshaped, axis=0)      # (n_phi, n_theta)
            max_mm_sky = np.max(mm_reshaped, axis=0)        # (n_phi, n_theta)
            median_mm_sky = np.median(mm_reshaped, axis=0)  # (n_phi, n_theta)
            
            # Get theta/phi grids from Cover_Sphere
            theta_vals = self.cover_forward_nn.theta  # (n_grid,)
            phi_vals = self.cover_forward_nn.phi      # (n_grid,)
            
            # Mean mismatch plot
            plt.figure() #plt.figure(figsize=(8, 6))
            # Transpose mean_mm_sky from (n_phi, n_theta) to (n_theta, n_phi) for plotting
            # so that theta is on Y-axis and phi is on X-axis
            PHI, THETA = np.meshgrid(phi_vals, theta_vals)  # (n_theta, n_phi)
            pcm = plt.pcolormesh(np.degrees(PHI), np.degrees(THETA), mean_mm_sky.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Mean $\mathfrak{m}$')
            plt.xlabel(r'$\phi$ (degrees)')
            plt.ylabel(r'$\theta$ (degrees)')
            plt.savefig(f'{plots_dir}/sxs_mm_vs_sky{suffix}.png', dpi=300)
            plt.savefig(f'{plots_dir}/sxs_mm_vs_sky{suffix}.pdf', dpi=300)
            plt.close()
            
            # Max mismatch plot
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(np.degrees(PHI), np.degrees(THETA), max_mm_sky.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Max $\mathfrak{m}$')
            plt.xlabel(r'$\phi$ (degrees)')
            plt.ylabel(r'$\theta$ (degrees)')
            plt.savefig(f'{plots_dir}/sxs_mm_max_vs_sky{suffix}.png', dpi=300)
            plt.close()
            
            # Median mismatch plot
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(np.degrees(PHI), np.degrees(THETA), median_mm_sky.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Median $\mathfrak{m}$')
            plt.xlabel(r'$\phi$ (degrees)')
            plt.ylabel(r'$\theta$ (degrees)')
            plt.savefig(f'{plots_dir}/sxs_mm_median_vs_sky{suffix}.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate sky mismatch plot: {e}")

    def plot_losses(self, train_losses, val_losses, save_dir=None, suffix=''):
        plots_dir = save_dir if save_dir is not None else self.config['output']['plots_dir']
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.figure()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"{plots_dir}/losses_summed{suffix}.png", dpi=300)
        plt.close()

    def plot_advanced_diagnostics(self, model, val_dl, suffix='', save_dir=None):
        """Generate advanced diagnostic plots to identify sources of high mismatch."""
        from scipy.stats import binned_statistic_2d
        
        plots_dir = save_dir if save_dir is not None else os.path.join(self.config['output']['plots_dir'], 'diagnostics')
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"Generating advanced diagnostic plots in {plots_dir}...")
        
        # Collect all validation data
        all_params = []
        all_sphere_mm = []
        all_mode_mm = []  # Per-mode mismatches
        
        model.eval()
        with torch.no_grad():
            for vparams, vwf in val_dl:
                # Raw predictions (per mode)
                raw_pred = model(vparams.to(self.device))  # (batch, n_modes, time)
                raw_target = vwf.to(self.device)           # (batch, n_modes, time)
                
                # Sphere-projected mismatch
                outputs_wave = self.cover_forward_nn(raw_pred).flatten(0, 2)
                vwf_wave = self.cover_forward_sxs(vwf).to(self.device).flatten(0, 2)
                sphere_mm = mymismatch(outputs_wave, vwf_wave)
                sphere_mm = torch.nan_to_num(sphere_mm)
                
                # Per-mode mismatch (one value per mode per sample)
                n_modes = raw_pred.shape[1]
                mode_mm = []
                for m in range(n_modes):
                    mm_m = mymismatch(raw_pred[:, m, :], raw_target[:, m, :])
                    mm_m = torch.nan_to_num(mm_m)
                    mode_mm.append(mm_m)
                mode_mm = torch.stack(mode_mm, dim=1)  # (batch, n_modes)
                
                all_params.append(vparams.cpu().numpy())
                # Mean sphere mm over sky positions per sample
                n_grid = self.cover_forward_nn.n
                sphere_mm_reshaped = sphere_mm.reshape(-1, n_grid * n_grid).mean(dim=1)
                all_sphere_mm.append(sphere_mm_reshaped.cpu().numpy())
                all_mode_mm.append(mode_mm.cpu().numpy())
        
        all_params = np.concatenate(all_params, axis=0)  # (N, 3): q, chi1, chi2
        all_sphere_mm = np.concatenate(all_sphere_mm, axis=0)  # (N,)
        all_mode_mm = np.concatenate(all_mode_mm, axis=0)  # (N, n_modes)
        
        q = all_params[:, 0]
        chi1 = all_params[:, 1]
        chi2 = all_params[:, 2]
        
        q_phys = 1.0 / q
        chi_eff = (chi1 + q * chi2) / (1.0 + q)
        
        # ===== 1. Mismatch vs chi1 =====
        plt.figure() #plt.figure(figsize=(8, 6))
        plt.scatter(chi1, all_sphere_mm, alpha=0.5, s=10)
        plt.xlabel(r'$\chi_1$')
        plt.ylabel(r'Mean $\mathfrak{m}$ (over sky)')
        plt.yscale('log')
        plt.savefig(f'{plots_dir}/mm_vs_chi1{suffix}.png', dpi=300)
        plt.close()
        
        # ===== 2. Mismatch vs chi2 =====
        plt.figure() #plt.figure(figsize=(8, 6))
        plt.scatter(chi2, all_sphere_mm, alpha=0.5, s=10)
        plt.xlabel(r'$\chi_2$')
        plt.ylabel(r'Mean $\mathfrak{m}$ (over sky)')
        plt.yscale('log')
        plt.savefig(f'{plots_dir}/mm_vs_chi2{suffix}.png', dpi=300)
        plt.close()
        
        # ===== 3. Per-mode mismatch histograms =====
        plt.figure(figsize=(10, 6))
        for m, mode in enumerate(self.modes):
            mm_mode = all_mode_mm[:, m]
            valid_mm = mm_mode[mm_mode > 0]
            if len(valid_mm) > 0:
                bins = np.logspace(np.log10(valid_mm.min()), np.log10(valid_mm.max()), 10)
                plt.hist(mm_mode, bins=bins, histtype='step', label=f'{mode}', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\mathfrak{m}$')
        plt.ylabel('Count')
        plt.legend(title=r'Mode ($\ell$,m)')
        plt.savefig(f'{plots_dir}/mm_per_mode_hist{suffix}.png', dpi=300)
        plt.close()
        
        # ===== 4. Per-mode box plot =====
        plt.figure() #plt.figure(figsize=(8, 6))
        mode_labels = [str(m) for m in self.modes]
        plt.boxplot([all_mode_mm[:, m] for m in range(len(self.modes))], label=mode_labels)
        plt.ylabel(r'$\mathfrak{m}$')
        plt.xlabel(r'Mode ($\ell$,m)')
        plt.yscale('log')
        plt.savefig(f'{plots_dir}/mm_per_mode_box{suffix}.png', dpi=300)
        plt.close()
        
        # ===== 5. 2D heatmap: q vs chi1 =====
        try:
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                q, chi1, all_sphere_mm, statistic='mean', bins=15
            )
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(x_edges, y_edges, stat.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Mean $\mathfrak{m}$')
            plt.xlabel('q')
            plt.ylabel(r'$\chi_1$')
            plt.savefig(f'{plots_dir}/mm_2d_q_chi1{suffix}.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate q vs chi1 heatmap: {e}")
        
        # ===== 6. 2D heatmap: chi1 vs chi2 =====
        try:
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                chi1, chi2, all_sphere_mm, statistic='mean', bins=15
            )
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(x_edges, y_edges, stat.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Mean $\mathfrak{m}$')
            plt.xlabel(r'$\chi_1$')
            plt.ylabel(r'$\chi_2$')
            plt.savefig(f'{plots_dir}/mm_2d_chi1_chi2{suffix}.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate chi1 vs chi2 heatmap: {e}")
        
        # ===== 7. 2D heatmap: q vs chi2 =====
        try:
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                q, chi2, all_sphere_mm, statistic='mean', bins=15
            )
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(x_edges, y_edges, stat.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Mean $\mathfrak{m}$')
            plt.xlabel('q')
            plt.ylabel(r'$\chi_2$')
            plt.savefig(f'{plots_dir}/mm_2d_q_chi2{suffix}.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate q vs chi2 heatmap: {e}")
        
        # ===== 8. 2D heatmap: q (inverted, >=1) vs chi_eff =====
        try:
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                q_phys, chi_eff, all_sphere_mm, statistic='mean', bins=15
            )
            plt.figure() #plt.figure(figsize=(8, 6))
            pcm = plt.pcolormesh(x_edges, y_edges, stat.T, 
                                  shading='auto', norm=plt.matplotlib.colors.LogNorm())
            plt.colorbar(pcm, label=r'Mean $\mathfrak{m}$')
            plt.xlabel('q')
            plt.ylabel(r'$\chi_{\mathrm{eff}}$')
            plt.savefig(f'{plots_dir}/mm_2d_q_chieff{suffix}.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate q vs chi_eff heatmap: {e}")
        
        print(f"Advanced diagnostics saved to {plots_dir}")
        # print params of largest mismatch
        max_idx = np.argmax(all_sphere_mm)
        print(f"Largest mismatch: {all_sphere_mm[max_idx]}")
        print(f"Parameters: {all_params[max_idx]}")



def main():
    parser = argparse.ArgumentParser(description="SXS HM Training")
    parser.add_argument('--config', type=str, default='scripts/config.yaml', help='Path to config file')
    parser.add_argument('--plot-only', action='store_true', help='Load saved model and generate plots without training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (default: uses models_dir/decoder_mode_summed.pt)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Ensure directories
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    
    # Random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading Dataset...")
    modes = [tuple(m) for m in config['modes']]

    
    ds = SXSDataset(config['data_path'], modes=modes)
    filt =  (ds.params_data[:,0] >= 1/8)*\
            (abs(ds.params_data[:,1]) <= 0.81)*\
            (abs(ds.params_data[:,2]) <= 0.81)*\
            (np.abs(ds.metadata['eccentricity'])<1e-2)*\
            (abs(ds.waveform_data[:,0,-1])<1e-1)
            # (abs(ds.waveform_data[:,0,:]).sum(axis=-1)<1000)

    print("###############################")
    print("Number of filtered events:",filt.sum())
    print("###############################")
    ds.waveform_data = abs(ds.waveform_data) * np.exp(1j * get_phases(ds.waveform_data).numpy())
    ds.waveform_data = ds.waveform_data[filt]
    ds.params_data = ds.params_data[filt]
    ds.data_len = len(ds.waveform_data)
    
    # Split
    idx = np.random.permutation(len(ds))
    split = int(len(ds) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]
    
    train_ds = MyDataset(X=ds.params_data[train_idx], y=ds.waveform_data[train_idx], device=device)
    val_ds = MyDataset(X=ds.params_data[val_idx], y=ds.waveform_data[val_idx], device=device)
    
    # Fix types (original script fixes)
    train_ds.X = train_ds.X.float()
    train_ds.y = torch.as_tensor(ds.waveform_data[train_idx]).to(device, dtype=torch.cfloat)
    val_ds.X = val_ds.X.float()
    val_ds.y = torch.as_tensor(ds.waveform_data[val_idx]).to(device, dtype=torch.cfloat)
    
    # Load Pretrained Decoders
    layers = config['model']['layers']
    decoders = []
    
    # Original script loop:
    # for mode in range(4): ... load ...
    # mapped modes (2,2), (3,3) etc.
    # config['modes'] has them.
    # We need a mode map for filenames or use the modes directly?
    # Original script:
    # mode_map = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
    # state_dict = torch.load(f'pretrain_files_SEOBNRv5HM/{mode_map[mode]}/models/decoder.pt', ...)
    
    for i, mode in enumerate(modes):
        path = config['model']['pretrain_path_template'].format(mode=mode)
        # Note: formatting tuple (2,2) into path string might depend on how python formats tuples.
        # Original: f'.../{mode_map[mode]}/...' where mode_map[mode] is (2,2).
        # string representation of (2, 2) is "(2, 2)".
        # CHECK if files exist with "(2, 2)" in name or just "2,2"?
        # Actually user might have folders like "(2, 2)".
        # Let's hope str(mode) works.
        
        state_dict = torch.load(path, map_location=device)
        # Handle compiled models
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        amp_basis = state_dict['amp_basis']
        amp_mean = state_dict['amp_mean']
        phase_basis = state_dict['phase_basis']
        phase_mean = state_dict['phase_mean']
        _model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=nn.GELU, device=device)
        _model.load_state_dict(state_dict)
        _model.float()
        _model.eval()
        decoders.append(_model)
    
    # Ensemble
    ens = DANSurEnsemble(decoders, modes, device=device)
    ens.float()
    
    # Create trainer (needed for validation and plotting)
    trainer = SXSTrainer(config, device)
    
    # Dataloaders
    bs_train = int(len(train_ds) * config['training'].get('batch_size_ratio', 1))
    bs_val = len(val_ds)  # Full batch val
    
    train_dl = MultiEpochsDataLoader(train_ds, batch_size=bs_train, shuffle=True)
    val_dl = MultiEpochsDataLoader(val_ds, batch_size=bs_val, shuffle=False)
    
    if args.plot_only:
        # Load saved model and generate plots
        model_path = args.model_path or f"{config['output']['models_dir']}/decoder_mode_summed.pt"
        print(f"Loading model from {model_path}...")
        ens.load_state_dict(torch.load(model_path, map_location=device))
        ens.eval()
        
        print("Running validation and generating plots...")
        with torch.no_grad():
            val_results = trainer.validate(ens, val_dl)
            mm_loss_valid = val_results['mm_loss_valid']
            print(f"Validation loss: {val_results['val_loss']:.4e}")
            print(f"Mean mismatch: {mm_loss_valid.mean().item():.4e}")
            print(f"Max mismatch: {mm_loss_valid.max().item():.4e}")
            
            trainer.plot_diagnostics(0, mm_loss_valid, val_results, ens, suffix='_loaded')
        
        # Generate advanced diagnostic plots
        trainer.plot_advanced_diagnostics(ens, val_dl, suffix='_loaded')
        
        print(f"Plots saved to {config['output']['plots_dir']}")
    else:
        # Normal training flow
        # Freeze logic
        print(ens.parameters())
        for param in ens.parameters():
            param.requires_grad = True
        
        # for param in ens.parameters():
        #     if param.shape[0] == 85:
        #         param.requires_grad = True

        # Optimizer
        optclass = topt.load_optimizer('adamw')
        optimizer = optclass(ens.parameters(), lr=float(config['training']['learning_rate'])) #type: ignore
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        
        # Train
        trainer.train(ens, optimizer, train_dl, val_dl, scheduler=scheduler)


if __name__ == "__main__":
    main()
