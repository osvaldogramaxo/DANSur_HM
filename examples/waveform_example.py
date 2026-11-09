"""
DANSur_HM — Minimal waveform generation example.

Loads the bundled DANSur_HM model and generates a time-domain gravitational
waveform including higher-order modes.

Supported parameter space:
  - Mass ratio:  1 <= q <= 8
  - Aligned spins:  |chi1z|, |chi2z| <= 0.8
  - Modes: (2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3)
"""

import numpy as np
from dansur import DANSur

# ── 1. Load the model (the trained weights ship with the package) ────────────
sur = DANSur()

# ── 2. Binary parameters ─────────────────────────────────────────────────────
q = 3.0       # mass ratio  m1/m2 >= 1
chi1z = 0.3   # dimensionless aligned spin of the heavier BH
chi2z = -0.1  # dimensionless aligned spin of the lighter BH

# ── 3. Dimensionless modes ───────────────────────────────────────────────────
# Time array in geometric units (M); covers ~4000 M before merger.
# With inclination=None the call returns (times, {(l, m): h_lm}, None),
# including the m < 0 modes.
times = np.arange(-3996, 100, 0.5)
t, h_modes, _ = sur(q=q, chiA0=chi1z, chiB0=chi2z, times=times)
print("Generated modes:", sorted(h_modes))

# ── 4. Strain in physical units ──────────────────────────────────────────────
# With an inclination the call returns the complex strain h = h+ - i hx.
# units='mks': masses in solar masses, distance in Mpc, times in seconds
# (merger at t = 0; samples outside the waveform are zero).
t_s, h, _ = sur(
    q=q, chiA0=chi1z, chiB0=chi2z,
    M=60.0, dist_mpc=500.0, units="mks", times=np.arange(-1.0, 0.03, 1 / 4096),
    inclination=np.pi / 4, phi_ref=0.0,
)
hp, hc = h.real, -h.imag
print(f"Peak |h|: {np.abs(h).max():.3e} over {t_s[-1] - t_s[0]:.2f} s")

# ── 5. Plot ──────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed — skipping plot.")

modes = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]
fig, axes = plt.subplots(len(modes) + 1, 1, figsize=(10, 14), sharex=False)
for ax, mode in zip(axes, modes):
    ax.plot(t, h_modes[mode].real, lw=0.8)
    ax.set_ylabel(rf"$h_{{{mode[0]}{mode[1]}}}$")
    ax.set_xlim(-1000, 100)
axes[-2].set_xlabel("Time [M]")
axes[-1].plot(t_s, hp, lw=0.8, label=r"$h_+$")
axes[-1].plot(t_s, hc, lw=0.8, label=r"$h_\times$")
axes[-1].set_xlim(-0.3, 0.05)
axes[-1].set_xlabel("Time [s]")
axes[-1].legend()
fig.suptitle(rf"DANSur_HM: $q={q}$, $\chi_{{1z}}={chi1z}$, $\chi_{{2z}}={chi2z}$")
plt.tight_layout()
plt.savefig("waveform_example.png", dpi=150)
print("Plot saved to waveform_example.png")
