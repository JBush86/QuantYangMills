# Spectral Threshold: Constructive Yang-Mills Mass Gap Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18158682.svg)](https://doi.org/10.5281/zenodo.18158682)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Author:** Jared C. Bush  
**Paper:** [A Non-Perturbative Lower Bound on the Spectrum of Euclidean Yang-Mills Theory](https://doi.org/10.5281/zenodo.18158682)

---

## Project Overview

This repository contains the **Lattice Monte Carlo simulations** and **Python reference implementations** used to validate the **Spectral Threshold** mechanism proposed in the associated research paper.

The core discovery is a rigorous information-theoretic bound for the **Yang-Mills Mass Gap**, defined by the Spectral Threshold identity:

$$I = E = \int (\mathcal{L} \cdot \omega \cdot \chi) dt$$

Where:
* $\mathcal{L}$ is the **Action Density** (Local field fluctuation).
* $\omega$ is the **Topological Weight** (Geometric non-triviality).
* $\chi$ is the **Spectral Coherence** (Stability against global action).

These scripts demonstrate that a non-zero mass gap ($\Delta > 0$) emerges naturally from the vacuum when the information density exceeds a critical threshold, effectively "freezing" quantum fluctuations into stable massive particles.

---

## Key Results

### 1. The Emergence of Mass
The simulation proves that sub-threshold fluctuations decay (massless glueballs), while super-threshold excitations persist as stable massive particles.

![Field Evolution](images/gauge_field_evolution.png)
*Figure 1: Top: Raw Gauge Field fluctuations. Bottom: The Information Field ($I$) showing stable particles forming only where the Spectral Threshold is crossed.*

### 2. Robustness & Scaling
We performed rigorous statistical sweeps across coupling constants ($\beta$) and energy budgets to validate the mechanism.

* **Continuum Limit:** $\Delta I_{crit}$ scales monotonically with $\beta$, consistent with Renormalization Group flow.
* **Linearity:** Total Information ($I$) scales linearly with Energy ($E$), satisfying the First Law of Thermodynamics ($I=E$).
* **Stability:** Growth exponents ($\mu$) are orders of magnitude smaller than decay exponents ($\lambda$), confirming particle stability.

![Beta Sweep](images/beta_sweep_results.png)
*Figure 2: Statistical validation of the Mass Gap threshold across different lattice parameters.*

---

## Repository Structure

| Script | Description | Complexity |
| :--- | :--- | :--- |
| **`QuantYangMills.py`** | **Core Logic.** Basic 1D implementation of the $L \cdot \omega \cdot \chi$ threshold. Visualizes the separation of noise vs. particle. | ★☆☆ |
| **`QuantYangMills2.py`** | **Energy Conservation.** Implements the "Energy Budget" to prove $I=E$ accounting. | ★★☆ |
| **`QuantYangMills3.py`** | **High-Res Simulation.** Runs on $N=128$ lattice to demonstrate continuum behavior. | ★★☆ |
| **`QuantYangMills4.py`** | **SU(3) Implementation.** Full non-Abelian gauge group simulation using Unitary matrices. | ★★★ |
| **`QuantYangMills5.py`** | **Statistical Framework.** Modular architecture for running parameter sweeps. | ★★★ |
| **`QuantYangMills6.py`** | **Final Validation.** The rigorous analysis suite. Runs $\beta$-sweeps, seed batches, and generates fit exponents. | ★★★ |

---

## Installation & Usage

### Prerequisites
* Python 3.9+
* `numpy`
* `matplotlib`
* `scipy`

### Quick Start
Clone the repository and run the primary simulation:

```bash
git clone [https://github.com/JBush86/Spectral-Threshold-Yang-Mills.git](https://github.com/JBush86/Spectral-Threshold-Yang-Mills.git)
cd Spectral-Threshold-Yang-Mills
pip install -r requirements.txt
python QuantYangMills.py
```

## Citation

### If you use this code or the Spectral Threshold framework in your research, please cite the original paper listed above.
Thank you!
