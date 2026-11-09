# DANSur_HM — Dual-stage Artificial Neural Surrogate (Now with Higher-Modes!)

A fast, accurate neural surrogate for gravitational waveforms from
quasi-circular, aligned-spin binary black hole mergers, including
higher-order spherical harmonic modes. Follow up to [arXiv:2412.06946](https://arxiv.org/abs/2412.06946), where only the (2,2) mode was considered.

Results published in: [arXiv:2609.03025](https://arxiv.org/abs/2609.03025)
[![DOI](https://zenodo.org/badge/1359189927.svg)](https://doi.org/10.5281/zenodo.24081382)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osvaldogramaxo/DANSur_HM/blob/main/examples/quickstart.ipynb)

**Try it in 30 seconds:**

```bash
pip install "dansur-hm @ git+https://github.com/osvaldogramaxo/DANSur_HM"
```

```python
from dansur import DANSur
t, h_modes, _ = DANSur()(q=3.0, chiA0=0.3, chiB0=-0.1)   # trained model is bundled
```

Or open the [quickstart notebook](examples/quickstart.ipynb) in Colab — no install needed.

## Overview

**DANSur_HM** is a two-stage neural network approximant:

1. **Stage 1 (Decoder)** — A PCA-based decoder network, pre-trained on
   ~1 million waveforms from `NRHybSur3dq8`, maps binary parameters to
   per-mode waveform coefficients.
2. **Stage 2 (Fine-tuning)** — The ensemble is fine-tuned directly on
   SXS numerical relativity waveforms using a sphere-projected mismatch loss.

**Supported parameter space:**
| Parameter | Range |
|---|---|
| Mass ratio *q* = m₁/m₂ | [1, 8] |
| Aligned spin χ₁z | [−0.8, 0.8] |
| Aligned spin χ₂z | [−0.8, 0.8] |
| Eccentricity | ≈ 0 (quasi-circular) |

**Included modes:** (2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3)

---

## Repository Structure

```
.
├── dansur/                          # Main Python package
│   ├── bilby_source.py             # bilby source model (dansur_binary_black_hole)
│   ├── surrogate/
│   │   ├── DANSur_HM_jitted.pt     # Trained TorchScript model (~41 MB), bundled
│   │   ├── sur_utils.py            # DANSur evaluator class (inference)
│   │   ├── waveform_generation.py  # Waveform generation utilities (22-mode)
│   │   └── waveform_generation_HM.py  # Waveform generation (higher-mode)
│   ├── training/
│   │   ├── sxs_hm.py              # Main HM training script (SXS fine-tuning)
│   │   ├── train_decoder_ensemble.py  # Decoder architecture + training loop
│   │   └── jit_model.py           # Export trained model to TorchScript
│   └── utils/
│       ├── nn.py                  # Neural network decoder, PCA layers, activations
│       ├── waveforms.py           # Phase unwrap, overlap/mismatch metrics, Cover_Sphere
│       ├── data.py                # SXSDataset, DataLoaders, normalizers, HDF5 helpers
│       ├── losses.py              # Spherical harmonic mismatch loss (HMLoss), ASDL1Loss
│       ├── training.py            # EarlyStopping, train/test loops, basis filters
│       ├── generation.py          # Approximant waveform generation (SEOBNR, Phenom)
│       └── helpers.py             # Logging, path handling, and plotting helpers
├── scripts/
│   ├── calculate_sxs_mismatches.py  # Evaluate model against SXS catalog
│   ├── bilby_run_injection.py       # GW parameter estimation (injection tests)
│   └── config.yaml                  # Training configuration
├── examples/
│   ├── quickstart.ipynb            # Colab notebook: modes, strain, batches, bilby
│   └── waveform_example.py         # Minimal usage example
├── pyproject.toml
└── LICENSE
```

---

## Setup

The base install covers waveform evaluation (torch, numpy, numba, gwsurrogate).
Extras add what the rest of the repository needs:

| Install | For |
|---|---|
| `pip install dansur-hm` | Evaluating waveforms |
| `pip install "dansur-hm[bilby]"` | The bilby source model (adds bilby, lalsuite) |
| `pip install "dansur-hm[train]"` | Training, evaluation and PE scripts in `scripts/` |

Until the package is on PyPI, install from GitHub:
`pip install "dansur-hm[bilby] @ git+https://github.com/osvaldogramaxo/DANSur_HM"`.

> **Note:** `gwsurrogate` is distributed as source only, so installation needs a C compiler
> (available on Linux, macOS with Xcode tools, and Colab).

### Development install

```bash
git clone https://github.com/osvaldogramaxo/DANSur_HM && cd DANSur_HM
uv sync --extra train      # or: pip install -e ".[train]"
```

---

## Usage

### Waveform evaluation (inference)

```python
import numpy as np
from dansur import DANSur

sur = DANSur()   # bundled model; DANSur(device="cuda") runs the network on a GPU

q, chi1z, chi2z = 3.0, 0.3, -0.1   # q = m1/m2 >= 1, aligned spins
times = np.arange(-3996, 100, 0.5)  # geometric units (M), peak at t = 0

# Modes: returns (times, {(l, m): complex h_lm}, None), including m < 0
t, h_modes, _ = sur(q=q, chiA0=chi1z, chiB0=chi2z, times=times)
print(h_modes[(3, 3)].shape)   # (8192,)

# Strain h = h+ - i hx at a given inclination, in physical units
t_s, h, _ = sur(q=q, chiA0=chi1z, chiB0=chi2z, M=60.0, dist_mpc=500.0, units="mks",
                times=np.arange(-1.0, 0.03, 1 / 4096), inclination=np.pi / 4, phi_ref=0.0)
hp, hc = h.real, -h.imag
```

**Batches:** pass arrays of `q` and spins as `(N, 3)` vectors with zero in-plane
components (e.g. `np.stack([zeros, zeros, chi1z], -1)`); outputs then have shape `(N, n_times)`.

**GPU:** with `DANSur(device="cuda")` the network and all post-processing run on the GPU,
in chunks of 1024 waveforms; only the final arrays are copied back. Strain for 10,000
waveforms takes ~1.4 s on a V100 vs ~7 s on 32 CPU cores. A single waveform takes ~20 ms
and is fastest on CPU, so prefer the CPU for one-at-a-time use such as bilby sampling.

See [`examples/waveform_example.py`](examples/waveform_example.py) for a runnable script
with plotting, and [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a notebook.

### Using DANSur_HM in bilby

Requires the `bilby` extra.

```python
import bilby
from dansur.bilby_source import dansur_binary_black_hole

waveform_generator = bilby.gw.WaveformGenerator(
    duration=4, sampling_frequency=2048,
    time_domain_source_model=dansur_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)
```

Use it like any other bilby waveform generator (sample in `chirp_mass`/`mass_ratio` or
`mass_1`/`mass_2`, with aligned spins `chi_1`, `chi_2`). Polarizations follow the LAL
convention. The waveform covers about 4000 M before merger, so the (2,2) mode starts at
roughly 18 Hz × (70 M☉ / M); pick the analysis minimum frequency above that.
To use a different model instance, call `dansur.bilby_source.model_gen_func(model, times, **params)`.

---

## Training Data Format

DANSur_HM relies on HDF5 files formatted for each of its two training stages:

### 1. Pretraining Dataset (Stage 1)

Used by `train_decoder_ensemble.py` to pretrain single-mode decoders on synthetic waveforms generated from baseline approximants (e.g., via `dansur.utils.generation.generate_dataset` using `NRHybSur3dq8`):

An HDF5 file containing:
- **`Waveforms`**: Complex array of shape `(N, n_modes, time_samples)` (`complex128`), with modes following the order: `[(2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3)]`.
- **`Parameters`**: Float array of shape `(N, 3)` (`float32`) containing binary parameters: `[q, chi_1z, chi_2z]`, where $q = m_1/m_2 \ge 1$.
  - Attribute: `Parameters.attrs['names'] = ['mass_ratio', 'chi_1', 'chi_2']`.

### 2. Numerical Relativity SXS Dataset (Stage 2 & Evaluation)

Used by `sxs_hm.py` (fine-tuning) and `calculate_sxs_mismatches.py` (evaluation), loaded via `dansur.utils.data.SXSDataset`:

An HDF5 file where each top-level key corresponds to a simulation ID (e.g., `SXS:BBH:0001`):

```
sxs_waveforms.h5
├── <simulation_id>/              # e.g. "SXS:BBH:0001"
│   ├── waveform/                 # Mode datasets
│   │   ├── (2,2)                 # 1D complex128 array of time-aligned samples
│   │   ├── (3,3)
│   │   ├── (2,1)
│   │   ├── (4,4)
│   │   ├── (3,2)
│   │   ├── (5,5)
│   │   └── (4,3)
│   ├── metadata/                 # Simulation physical parameters
│   │   ├── mass_ratio            # Float: q = m1/m2 >= 1
│   │   ├── dimensionless_spin1   # Array of floats: [s1x, s1y, s1z]
│   │   ├── dimensionless_spin2   # Array of floats: [s2x, s2y, s2z]
│   │   ├── chi1_perp             # Float: in-plane spin magnitude for BH 1
│   │   ├── chi2_perp             # Float: in-plane spin magnitude for BH 2
│   │   └── eccentricity          # Float: residual orbital eccentricity
│   └── TS_metadata/              # Time-series metadata (sample times, dt, etc.)
```

---

## Training

### 1. Pretraining

Pretrain individual mode decoders on a large dataset (e.g., from `NRHybSur3dq8`).
Each decoder learns to reconstruct a single spherical harmonic mode from a PCA basis:

```bash
# Run for each mode index 0–6 (corresponding to (2,2),(3,3),(2,1),(4,4),(3,2),(5,5),(4,3))
python -m dansur.training.train_decoder_ensemble --mode 0
```

### 2. HM Fine-tuning on SXS NR data

Fine-tune the full multi-mode ensemble on SXS numerical relativity waveforms
using a sphere-projected mismatch loss:

```bash
python -m dansur.training.sxs_hm \
    --config scripts/config.yaml \
    --sxs-file /path/to/sxs_waveforms.h5
```

Key configuration options are in [`scripts/config.yaml`](scripts/config.yaml).

### 3. Export to TorchScript (JIT)

After training, export the model for fast inference:

```bash
python -m dansur.training.jit_model \
    --config scripts/config.yaml \
    --output-path dansur/surrogate/DANSur_HM_jitted.pt
```

---

## Evaluation

Compute mismatches of DANSur_HM and NRHybSur3dq8 against the SXS catalog:

```bash
python scripts/calculate_sxs_mismatches.py \
    --sxs-file /path/to/sxs_waveforms.h5 \
    --out-dir plots/
```

---

## Citation

If you use DANSur_HM in your research, please cite:

```bibtex
@article{gramaxo2026dansur,
  title  = {DANSur_HM: Modularly incorporating higher modes in a deep learning based gravitational-wave surrogate},
  author = {Gramaxo Freitas, Osvaldo and others},
  year   = {2026},
  eprint = {2609.03025},
  archivePrefix = {arXiv},
}
```

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
