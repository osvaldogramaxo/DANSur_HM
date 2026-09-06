"""
DANSur_HM — Minimal waveform generation example.

This script demonstrates how to load the trained DANSur_HM model and generate
a gravitational waveform in the time domain including higher-order modes.

Supported parameter space:
  - Mass ratio:  1 <= q <= 8
  - Aligned spins:  |chi1z|, |chi2z| <= 0.8
  - Modes: (2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3)
"""

import numpy as np
from dansur.surrogate import DANSur

# ── 1. Load the model ────────────────────────────────────────────────────────
MODEL_PATH = "models/DANSur_HM_jitted.pt"  # path relative to repo root

modes = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]
sur = DANSur(path=MODEL_PATH, modes_list=modes)

# ── 2. Set binary parameters ──────────────────────────────────────────────────
q      = 3.0    # mass ratio  m1/m2 >= 1
chi1z  = 0.3    # dimensionless aligned spin of the heavier BH
chi2z  = -0.1   # dimensionless aligned spin of the lighter BH

# Physical parameters for strain output
total_mass = 60.0        # solar masses
distance   = 500.0       # Mpc
inclination = np.pi / 4  # inclination angle (radians)

# Time array in geometric units (M); covers ~4000 M before merger
times = np.arange(-3996, 100, 0.5)

# ── 3. Generate the dimensionless mode amplitudes ────────────────────────────
#   Returns: (hp_dimless, hc_dimless, h_modes_dict)
#   where h_modes_dict[(l,m)] is the complex waveform for each mode.
hp, hc, h_modes = sur(
    q=q,
    chiA0=chi1z,
    chiB0=chi2z,
    inclination=inclination,
    mode_list=modes,
    times=times,
)

print("Generated modes:", list(h_modes.keys()))
print(f"  (2,2) mode shape: {h_modes[(2,2)].shape}")
print(f"  (3,3) peak amplitude: {np.max(np.abs(h_modes[(3,3)])):.4e}")

# ── 4. Quick plot ─────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 12), sharex=True)
    for ax, (l, m) in zip(axes, modes):
        ax.plot(times, h_modes[(l, m)].real, lw=0.8)
        ax.set_ylabel(rf"$h_{{{l}{m}}}$", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [M]")
    fig.suptitle(
        rf"DANSur_HM: $q={q}$, $\chi_{{1z}}={chi1z}$, $\chi_{{2z}}={chi2z}$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("examples/waveform_example.png", dpi=150)
    print("Plot saved to examples/waveform_example.png")
    plt.show()
except ImportError:
    print("matplotlib not available — skipping plot.")
