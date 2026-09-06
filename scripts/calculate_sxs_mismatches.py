#!/usr/bin/env python
"""
Calculate mismatches of DANSur and NRHybSur3dq8 against SXS waveforms, 
and directly compare the physical implementations of DANSur and NRHybSur3dq8.
"""

import os
import sys
import numpy as np
import h5py
import torch
import gwsurrogate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure workspace root is in path

from dansur.surrogate.sur_utils import DANSur
from dansur.utils.sxs_utils import SXSDataset
from dansur.utils.utils import Cover_Sphere, get_phases

def shift_waveform(h, shift):
    """Shift a complex waveform by an integer index shift, padding with 0."""
    if shift == 0:
        return h.copy()
    elif shift > 0:
        return np.pad(h, (shift, 0))[:-shift]
    else:
        return np.pad(h, (0, -shift))[-shift:]

def calculate_mismatch(h1, h2):
    """Calculate the overlap mismatch (1 - overlap) between two complex waveforms.
    
    Optimizes over a constant phase shift.
    Works with both 1D arrays and 2D arrays (across the last axis).
    """
    axis = -1
    inner = np.sum(h1 * np.conj(h2), axis=axis)
    norm1 = np.sqrt(np.sum(np.abs(h1)**2, axis=axis))
    norm2 = np.sqrt(np.sum(np.abs(h2)**2, axis=axis))
    
    # Avoid division by zero
    denom = norm1 * norm2
    overlap = np.zeros_like(inner, dtype=float)
    valid = denom > 0
    
    if np.isscalar(inner):
        if denom > 0:
            overlap = np.abs(inner) / denom
        return 1.0 - overlap
    else:
        overlap[valid] = np.abs(inner[valid]) / denom[valid]
        return 1.0 - overlap

def find_optimal_shift(h_target_22, h_source_22, max_search=15):
    """Find the optimal index shift to align the source (2,2) mode to the target (2,2) mode."""
    peak_target = np.argmax(np.abs(h_target_22))
    peak_source = np.argmax(np.abs(h_source_22))
    init_shift = peak_target - peak_source
    
    best_mismatch = 1.0
    best_shift = init_shift
    
    for s in range(init_shift - max_search, init_shift + max_search + 1):
        h_shifted = shift_waveform(h_source_22, s)
        m = calculate_mismatch(h_target_22, h_shifted)
        if m < best_mismatch:
            best_mismatch = m
            best_shift = s
            
    return best_shift

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate mismatches against SXS dataset.")
    parser.add_argument("--sxs-file", type=str, default="../sxs/sxs_waves_allmodes_8192.h5", help="Path to SXS HDF5 file.")
    parser.add_argument("--model-path", type=str, default="models/rolling/DANSur_HM_jitted.pt", help="Path to JIT model.")
    parser.add_argument("--out-dir", type=str, default="plots", help="Output directory.")
    parser.add_argument("--num-waveforms", type=int, default=None, help="Limit number of waveforms (for testing).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Define modes
    modes = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]
    mode_map = {i: m for i, m in enumerate(modes)}
    
    # 2. Load SXS Dataset
    print(f"Loading SXS dataset from {args.sxs_file}...")
    ds = SXSDataset(args.sxs_file, modes=modes)
    
    # Apply standard filters for aligned-spin and quasi-circular
    print("Applying filters (q <= 8, spins <= 0.81, eccentricity < 0.01)...")
    filt = (ds.params_data[:, 0] >= 1/8.0) * \
           (np.abs(ds.params_data[:, 1]) <= 0.81) * \
           (np.abs(ds.params_data[:, 2]) <= 0.81) * \
           (np.abs(ds.metadata['eccentricity']) < 1e-2) * \
           (np.abs(ds.waveform_data[:, 0, -1]) < 1e-1)
           
    filtered_indices = np.where(filt)[0]
    num_filtered = len(filtered_indices)
    print(f"Filtered {num_filtered} / {len(ds)} waveforms matching validity criteria.")
    
    if args.num_waveforms is not None:
        filtered_indices = filtered_indices[:args.num_waveforms]
        num_filtered = len(filtered_indices)
        print(f"Limiting to first {num_filtered} waveforms for processing.")
        
    # Phase-align target SXS modes to start at phase 0 (matching sxs_hm.py preprocessing)
    print("Phase-aligning SXS target modes...")
    # Keep copy of raw modes for potential raw comparisons
    h_sxs_raw_dataset = ds.waveform_data.copy()
    ds.waveform_data = abs(ds.waveform_data) * np.exp(1j * get_phases(torch.tensor(ds.waveform_data)).numpy())
        
    # 3. Load Models
    print(f"Loading DANSur from {args.model_path}...")
    dansur = DANSur(path=args.model_path, modes_list=modes)
    
    print("Loading NRHybSur3dq8...")
    sur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    
    # Initialize Cover_Sphere
    cover_sphere = Cover_Sphere(mode_map, n=10, infer_neg_m=True, device='cpu')
    cover_sphere.sylm_values = cover_sphere.sylm_values.to(torch.complex64)
    if hasattr(cover_sphere, 'neg_m_l_coeffs'):
        cover_sphere.neg_m_l_coeffs = cover_sphere.neg_m_l_coeffs.to(torch.complex64)
        
    # 4. Prepare storage for results
    results = {
        'keys': [],
        'q': [],
        'chi1z': [],
        'chi2z': [],
        # Part 1: Model vs SXS (Phase-Aligned)
        'dansur_sphere_mms': [],
        'nrsur_sphere_mms': [],
        'all_dansur_sphere_mms_flat': [],
        'all_nrsur_sphere_mms_flat': [],
        # Part 2: DANSur vs NRHybSur (Physical, no get_phase)
        'dansur_vs_nrsur_sphere_mms_raw': [],
        'dansur_vs_nrsur_sphere_mms_aligned': [],
        'all_dansur_vs_nrsur_sphere_mms_raw_flat': [],
        'all_dansur_vs_nrsur_sphere_mms_aligned_flat': []
    }
    
    times = np.arange(-3996, 100, 0.5)
    processed_count = 0
    
    # 5. Process loop
    print("Calculating mismatches...")
    for idx in tqdm(filtered_indices):
        key = ds.waveforms[idx]
        q_in = ds.params_data[idx, 0]
        q = 1.0 / q_in
        chi1z = ds.params_data[idx, 1]
        chi2z = ds.params_data[idx, 2]
        
        # Get target phase-aligned SXS modes
        h_sxs_aligned = ds.waveform_data[idx] # (7, 8192)
        
        # Evaluate raw phase-aligned DANSur predictions directly from JIT model
        x = torch.tensor([[q_in, chi1z, chi2z]], dtype=torch.float32)
        try:
            with torch.no_grad():
                h_dan_raw = dansur._sur_dimless(x).squeeze(0).cpu().numpy() # (7, 8192)
        except Exception as e:
            print(f"\nWarning: Skipping waveform {key} due to JIT evaluation error: {e}")
            continue
            
        # Evaluate physical waveforms for both models
        try:
            # DANSur physical
            _, h_dansur_dict, _ = dansur(q=q, chiA0=chi1z, chiB0=chi2z, inclination=None, mode_list=modes, times=times, skip_param_checks=True)
            h_dan_phys = np.stack([h_dansur_dict[m] for m in modes], axis=0)
            
            # NRHybSur physical
            _, h_sur_phys_dict, _ = sur(q=q, chiA0=[0.0, 0.0, chi1z], chiB0=[0.0, 0.0, chi2z], inclination=None, mode_list=modes, times=times, f_low=0.0)
            h_sur_phys = np.stack([h_sur_phys_dict[m] for m in modes], axis=0)
        except Exception as e:
            print(f"\nWarning: Skipping waveform {key} due to evaluation error: {e}")
            continue
            
        # Phase-align NRHybSur3dq8 modes to start at phase 0
        h_sur_aligned = abs(h_sur_phys) * np.exp(1j * get_phases(torch.tensor(h_sur_phys)).numpy())
        
        # Store metadata
        results['keys'].append(key)
        results['q'].append(q)
        results['chi1z'].append(chi1z)
        results['chi2z'].append(chi2z)
        processed_count += 1
        
        # Project Phase-Aligned waveforms to Sphere
        with torch.no_grad():
            sphere_sxs = cover_sphere(torch.as_tensor(h_sxs_aligned, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            sphere_dan = cover_sphere(torch.as_tensor(h_dan_raw, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            sphere_sur = cover_sphere(torch.as_tensor(h_sur_aligned, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            
        sphere_sxs_flat = sphere_sxs.reshape(-1, 8192)
        sphere_dan_flat = sphere_dan.reshape(-1, 8192)
        sphere_sur_flat = sphere_sur.reshape(-1, 8192)
        
        mms_dan_sphere = calculate_mismatch(sphere_sxs_flat, sphere_dan_flat)
        mms_sur_sphere = calculate_mismatch(sphere_sxs_flat, sphere_sur_flat)
        
        results['dansur_sphere_mms'].append(np.mean(mms_dan_sphere))
        results['nrsur_sphere_mms'].append(np.mean(mms_sur_sphere))
        results['all_dansur_sphere_mms_flat'].extend(mms_dan_sphere)
        results['all_nrsur_sphere_mms_flat'].extend(mms_sur_sphere)
        
        # --- Part 2: DANSur vs NRHybSur3dq8 (Physical, no get_phase) ---
        # Find best shift to align their peaks based on (2,2) mode
        shift = find_optimal_shift(h_sur_phys_dict[(2,2)], h_dansur_dict[(2,2)])
        
        # Align DANSur physical output to NRHybSur physical output
        h_dan_phys_aligned = np.stack([shift_waveform(h_dansur_dict[m], shift) for m in modes], axis=0)
        
        # Project to Sphere
        with torch.no_grad():
            sphere_sur_phys = cover_sphere(torch.as_tensor(h_sur_phys, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            sphere_dan_phys_raw = cover_sphere(torch.as_tensor(h_dan_phys, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            sphere_dan_phys_align = cover_sphere(torch.as_tensor(h_dan_phys_aligned, dtype=torch.complex64).unsqueeze(0)).squeeze(0).numpy()
            
        sphere_sur_phys_flat = sphere_sur_phys.reshape(-1, 8192)
        sphere_dan_phys_raw_flat = sphere_dan_phys_raw.reshape(-1, 8192)
        sphere_dan_phys_align_flat = sphere_dan_phys_align.reshape(-1, 8192)
        
        mms_model_raw = calculate_mismatch(sphere_sur_phys_flat, sphere_dan_phys_raw_flat)
        mms_model_align = calculate_mismatch(sphere_sur_phys_flat, sphere_dan_phys_align_flat)
        
        results['dansur_vs_nrsur_sphere_mms_raw'].append(np.mean(mms_model_raw))
        results['dansur_vs_nrsur_sphere_mms_aligned'].append(np.mean(mms_model_align))
        results['all_dansur_vs_nrsur_sphere_mms_raw_flat'].extend(mms_model_raw)
        results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'].extend(mms_model_align)
        
    # Convert lists to arrays
    for k in ['all_dansur_sphere_mms_flat', 'all_nrsur_sphere_mms_flat',
              'all_dansur_vs_nrsur_sphere_mms_raw_flat', 'all_dansur_vs_nrsur_sphere_mms_aligned_flat',
              'dansur_sphere_mms', 'nrsur_sphere_mms',
              'dansur_vs_nrsur_sphere_mms_raw', 'dansur_vs_nrsur_sphere_mms_aligned']:
        results[k] = np.array(results[k])
        
    # 6. Save results to npz
    save_path = os.path.join(args.out_dir, "sxs_mismatches.npz")
    np.savez(save_path, 
             keys=np.array(results['keys']),
             q=np.array(results['q']),
             chi1z=np.array(results['chi1z']),
             chi2z=np.array(results['chi2z']),
             dansur_sphere_mms=results['dansur_sphere_mms'],
             nrsur_sphere_mms=results['nrsur_sphere_mms'],
             all_dansur_sphere_mms_flat=results['all_dansur_sphere_mms_flat'],
             all_nrsur_sphere_mms_flat=results['all_nrsur_sphere_mms_flat'],
             dansur_vs_nrsur_sphere_mms_raw=results['dansur_vs_nrsur_sphere_mms_raw'],
             dansur_vs_nrsur_sphere_mms_aligned=results['dansur_vs_nrsur_sphere_mms_aligned'],
             all_dansur_vs_nrsur_sphere_mms_raw_flat=results['all_dansur_vs_nrsur_sphere_mms_raw_flat'],
             all_dansur_vs_nrsur_sphere_mms_aligned_flat=results['all_dansur_vs_nrsur_sphere_mms_aligned_flat']
    )
    print(f"Results saved to {save_path}")
    
    # 7. Print summary statistics table
    summary_txt_path = os.path.join(args.out_dir, "sxs_mismatches_summary.txt")
    with open(summary_txt_path, "w") as sf:
        def write_and_print(text):
            print(text)
            sf.write(text + "\n")
            
        write_and_print("\n" + "="*80)
        write_and_print(f"SXS SPHERE-PROJECTED MISMATCH STATS (N={processed_count})")
        write_and_print("="*80)
        
        # Model vs SXS stats
        med_dan = np.median(results['dansur_sphere_mms'])
        med_sur = np.median(results['nrsur_sphere_mms'])
        med_dan_flat = np.median(results['all_dansur_sphere_mms_flat'])
        med_sur_flat = np.median(results['all_nrsur_sphere_mms_flat'])
        max_dan = np.max(results['dansur_sphere_mms'])
        max_sur = np.max(results['nrsur_sphere_mms'])
        max_dan_flat = np.max(results['all_dansur_sphere_mms_flat'])
        max_sur_flat = np.max(results['all_nrsur_sphere_mms_flat'])
        
        write_and_print(f"{'Metric':<30} | {'DANSur':<15} | {'NRHybSur3dq8':<15}")
        write_and_print("-"*70)
        write_and_print(f"{'Median Sphere-Avg (vs SXS)':<30} | {med_dan:<15.4e} | {med_sur:<15.4e}")
        write_and_print(f"{'Median Point-by-Point (vs SXS)':<30} | {med_dan_flat:<15.4e} | {med_sur_flat:<15.4e}")
        write_and_print(f"{'Max Sphere-Avg (vs SXS)':<30} | {max_dan:<15.4e} | {max_sur:<15.4e}")
        write_and_print(f"{'Max Point-by-Point (vs SXS)':<30} | {max_dan_flat:<15.4e} | {max_sur_flat:<15.4e}")
        write_and_print("="*80)
        
        # Model vs Model stats
        write_and_print("\n" + "="*80)
        write_and_print("DANSur vs NRHybSur3dq8 DIRECT SPHERE COMPARISON (NO GET_PHASE)")
        write_and_print("="*80)
        
        med_direct_raw = np.median(results['dansur_vs_nrsur_sphere_mms_raw'])
        med_direct_align = np.median(results['dansur_vs_nrsur_sphere_mms_aligned'])
        med_direct_raw_flat = np.median(results['all_dansur_vs_nrsur_sphere_mms_raw_flat'])
        med_direct_align_flat = np.median(results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'])
        max_direct_raw = np.max(results['dansur_vs_nrsur_sphere_mms_raw'])
        max_direct_align = np.max(results['dansur_vs_nrsur_sphere_mms_aligned'])
        max_direct_raw_flat = np.max(results['all_dansur_vs_nrsur_sphere_mms_raw_flat'])
        max_direct_align_flat = np.max(results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'])
        
        write_and_print(f"{'Metric':<30} | {'Raw (no shift)':<15} | {'Peak-Aligned':<15}")
        write_and_print("-"*70)
        write_and_print(f"{'Median Sphere-Avg':<30} | {med_direct_raw:<15.4e} | {med_direct_align:<15.4e}")
        write_and_print(f"{'Median Point-by-Point':<30} | {med_direct_raw_flat:<15.4e} | {med_direct_align_flat:<15.4e}")
        write_and_print(f"{'Max Sphere-Avg':<30} | {max_direct_raw:<15.4e} | {max_direct_align:<15.4e}")
        write_and_print(f"{'Max Point-by-Point':<30} | {max_direct_raw_flat:<15.4e} | {max_direct_align_flat:<15.4e}")
        write_and_print("="*80)
        
    print(f"Text summary saved to {summary_txt_path}")
    
    # 8. Plot histograms & CDFs
    print("Generating comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Model vs SXS (Phase-Aligned)
    ax = axes[0]
    valid_dan = results['all_dansur_sphere_mms_flat'][results['all_dansur_sphere_mms_flat'] > 0]
    valid_sur = results['all_nrsur_sphere_mms_flat'][results['all_nrsur_sphere_mms_flat'] > 0]
    min_val = min(valid_dan.min(), valid_sur.min())
    max_val = max(valid_dan.max(), valid_sur.max())
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    
    ax.hist(results['all_dansur_sphere_mms_flat'], bins=bins, histtype='step', label="DANSur (vs SXS)", color='blue', linewidth=1.5)
    ax.hist(results['all_nrsur_sphere_mms_flat'], bins=bins, histtype='step', label="NRHybSur3dq8 (vs SXS)", color='orange', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathfrak{M}$')
    ax.set_ylabel('Count')
    ax.set_title('SXS Sphere-Projected Mismatches (Phase-Aligned)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Plot 2: Model vs Model (Physical)
    ax = axes[1]
    sorted_dan = np.sort(results['all_dansur_sphere_mms_flat'])
    sorted_sur = np.sort(results['all_nrsur_sphere_mms_flat'])
    y = np.linspace(0, 1, len(sorted_dan))
    ax.plot(sorted_dan, y, label="DANSur (vs SXS)", color='blue', lw=2)
    ax.plot(sorted_sur, y, label="NRHybSur3dq8 (vs SXS)", color='orange', lw=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mathfrak{M}$')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of SXS Mismatches')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "sxs_mismatches_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # New Plot: Direct DANSur vs NRHybSur comparison (no get_phase)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograms
    ax = axes[0]
    valid_raw = results['all_dansur_vs_nrsur_sphere_mms_raw_flat'][results['all_dansur_vs_nrsur_sphere_mms_raw_flat'] > 0]
    valid_align = results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'][results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'] > 0]
    min_val = min(valid_raw.min(), valid_align.min())
    max_val = max(valid_raw.max(), valid_align.max())
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    
    ax.hist(results['all_dansur_vs_nrsur_sphere_mms_raw_flat'], bins=bins, histtype='step', label="Raw (No alignment shift)", color='red', linewidth=1.5)
    ax.hist(results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'], bins=bins, histtype='step', label="Peak-Aligned", color='green', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathfrak{M}$')
    ax.set_ylabel('Count')
    ax.set_title('DANSur vs NRHybSur3dq8 Direct Sphere Mismatches')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # CDFs
    ax = axes[1]
    sorted_raw = np.sort(results['all_dansur_vs_nrsur_sphere_mms_raw_flat'])
    sorted_align = np.sort(results['all_dansur_vs_nrsur_sphere_mms_aligned_flat'])
    y = np.linspace(0, 1, len(sorted_raw))
    ax.plot(sorted_raw, y, label="Raw (No alignment shift)", color='red', lw=2)
    ax.plot(sorted_align, y, label="Peak-Aligned", color='green', lw=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mathfrak{M}$')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Direct Model Comparison')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    direct_plot_path = os.path.join(args.out_dir, "dansur_vs_nrsur_sphere_mismatches.png")
    plt.savefig(direct_plot_path, dpi=300)
    plt.close()
    print(f"Direct comparison plot saved to {direct_plot_path}")

if __name__ == "__main__":
    main()
