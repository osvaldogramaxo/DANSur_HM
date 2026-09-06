# DANSur_HM — Dual-stage Artificial Neural Surrogate (Higher-Mode)

A fast, accurate neural surrogate for gravitational waveforms from
quasi-circular, aligned-spin binary black hole mergers, including
higher-order spherical harmonic modes.

Results published in: [arXiv:2609.03025](https://arxiv.org/abs/2609.03025)

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
│   ├── surrogate/
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
├── models/
│   └── DANSur_HM_jitted.pt         # Trained TorchScript model (~41 MB)
├── examples/
│   └── waveform_example.py         # Minimal usage example
├── pyproject.toml
└── LICENSE
```

---

## Setup

### Using `uv` (recommended)

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Activate
source .venv/bin/activate
```

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note:** `lalsuite` installation may require system-level LIGO dependencies.
> See the [LALSuite documentation](https://lscsoft.docs.ligo.org/lalsuite/) for details.

---

## Usage

### Waveform evaluation (inference)

```python
import numpy as np
from dansur.surrogate import DANSur

modes = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)]
sur = DANSur(path="models/DANSur_HM_jitted.pt", modes_list=modes)

# Binary parameters
q     = 3.0   # mass ratio (m1/m2 >= 1)
chi1z = 0.3   # spin of heavier BH
chi2z = -0.1  # spin of lighter BH

# Time array in geometric units (M)
times = np.arange(-3996, 100, 0.5)

# Generate waveform (returns: hp, hc, h_modes dict)
hp, hc, h_modes = sur(
    q=q, chiA0=chi1z, chiB0=chi2z,
    inclination=np.pi/4, mode_list=modes, times=times,
)
print(h_modes[(2, 2)].shape)  # e.g. (8192,)
```

See [`examples/waveform_example.py`](examples/waveform_example.py) for a
complete runnable script with plotting.

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
    --output-path models/DANSur_HM_jitted.pt
```

---

## Evaluation

Compute mismatches of DANSur_HM and NRHybSur3dq8 against the SXS catalog:

```bash
python scripts/calculate_sxs_mismatches.py \
    --sxs-file /path/to/sxs_waveforms.h5 \
    --model-path models/DANSur_HM_jitted.pt \
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
