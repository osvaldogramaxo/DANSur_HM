"""
DANSur: Dual-stage Artificial Neural Surrogate for gravitational wave signals.

This package provides the DANSur_HM waveform approximant, supporting higher-order
spherical harmonic modes (2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3) for
quasi-circular, aligned-spin binary black hole systems with mass ratio q ≤ 8.

Typical usage:
    from dansur import DANSur
    sur = DANSur()  # loads the bundled trained model

Submodules with extra dependencies are imported on demand:
    dansur.bilby_source  (pip install "dansur-hm[bilby]")
    dansur.training, dansur.utils  (pip install "dansur-hm[train]")
"""

from .surrogate import DANSur

__all__ = ['DANSur']
