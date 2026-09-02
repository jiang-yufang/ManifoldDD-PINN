# Domain Decomposition Methods with Physics-Informed Neural Networks for Elliptic Equations on Manifolds

**Official implementation of the paper:**

> *Domain Decomposition Methods with Physics-Informed Neural Networks for Elliptic Equations on Manifolds* \
> Authors: Yufang Jiang(江郁芳), Lizhen Qin(秦理真), Feng Wang(王锋)

[![arXiv](https://img.shields.io/badge/arXiv-2607.04285-b31b1b.svg)](https://arxiv.org/abs/2607.04285)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2607.04285-blue.svg)](https://doi.org/10.48550/arXiv.2607.04285)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[English](README.md) | [简体中文](README.zh-CN.md)**

The companion paper (the same PDF included in this repository, `DDM_PINN_arXiv.pdf`) proposes two numerical **domain decomposition methods (DDMs)** for elliptic equations on compact Riemannian manifolds, based on **physics-informed neural networks (PINNs)**. This repository contains the complete PyTorch implementation used to produce the numerical results in **Section 4** of the paper.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
  - [Commands](#commands)
  - [Experiment Configuration](#experiment-configuration)
  - [Outputs and Results](#outputs-and-results)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)

---

## Overview

We solve the model problem

```
-Δu + b·u = f   on a d-dimensional compact Riemannian manifold M,
```

where Δ is the Laplace–Beltrami operator. A manifold is decomposed into finitely many overlapping subdomains, each lying in a local coordinate chart, and a PINN is trained on each subdomain to approximate the local solution. The subdomain problems are coupled by an iterative procedure — **sequential** or **parallel** with a partition of unity (see Algorithms 3.1 and 3.2 in the paper). When the manifold is decomposed into two subdomains, the two algorithms both reduce to the Schwarz Alternating Method (see Algorithm 3.3 in the paper).

The methods are validated numerically on four manifolds with or without boundary, in dimensions ranging from **5 to 10**:

| Manifold | Dimension | Boundary | Paper section |
|---|---|---|---|
| S⁵ (unit sphere in ℝ⁶) | 5 | no | §4.1 |
| S¹⁰ (unit sphere in ℝ¹¹) | 10 | no | §4.2 |
| B³ × S³ (product manifold) | 6 | yes (S² × S³) | §4.3 |
| CP³ (complex projective space) | 6 | no | §4.4 & §4.5 |

## Features

- **Two DDM variants on manifolds**, both implemented from scratch with PyTorch:
  - Sequential DDM — Algorithm 3.1 of the paper.
  - Parallel DDM with a partition of unity — Algorithm 3.2 of the paper.
- **The Specific Riemannian manifolds**:
  - S⁵ / S¹⁰ with standard metric,
  - B³ × S³ with standard metric,
  - CP³ with Fubini-Study metric.
- **Automatic device selection** — CUDA GPU when available, CPU otherwise.
- **Reproducible statistics** — every main script runs **5 independent trials** with random initialization and reports the mean and standard deviation of the relative L² errors at outer steps 0, 5, 10, 15, 20 and 100.
- **Automatic plots and logs** — relative-error convergence curves, ‖uⁿ − u^∞‖ curves, per-trial CSV logs and a statistics CSV are generated automatically.
- **Flexible network options** — plain fully-connected PINN or a residual (ResNet-style) variant; tunable depth, width, activation, optimizer, learning-rate schedule, sampling rate and loss weighting.

## Repository Structure

```
.
├── DDM_PINN_arXiv.pdf            # The companion paper (arXiv preprint)
├── requirements.txt              # Python dependencies
├── README.md                     # This file (English)
├── README.zh-CN.md               # README (Simplified Chinese)
├── LICENSE                       # MIT license
│
├── 4.1exp/                       # §4.1  Numerical experiments on S⁵
│   ├── HdDDM1.py                 #   Main program (Schwarz Alternating Method, two subdomains)
│   ├── pinn_models1.py           #   Training class, spherical Laplacian, losses
│   ├── basic_functions.py        #   Network architectures & sampling on balls/spheres
│   ├── test_functions.py         #   Error evaluation & plotting helpers
│   ├── data_processing.py        #   Mean/std statistics at steps 0,5,10,15,20,100
│   ├── record_result.py          #   CSV result logging
│   ├── error_curve/              #   Output: relative L² error curves
│   ├── un-u_inf_curve/           #   Output: ‖uⁿ − u^∞‖ curves
│   ├── result/                   #   Output: CSV logs & statistics
│   ├── models/                   #   Output: trained model checkpoints
│   ├── history_preds/            #   Output: saved predictions
│   └── history_error/            #   Output: saved error histories
│
├── 4.2exp/                       # §4.2  Numerical experiments on S¹⁰
│   └── ...                       #   Same layout as 4.1exp (main program: HdDDM1.py)
│
├── 4.3exp/                       # §4.3  Numerical experiments on B³ × S³
│   ├── HdDDM2.py                 #   Main program (Schwarz Alternating Method, two subdomains)
│   └── ...                       #   Same layout as 4.1exp
│
├── 4.4exp/                       # §4.4  Sequential DDM on CP³
│   ├── HdDDM3.py                 #   Main program (sequential DDM, four subdomains)
│   └── ...                       #   Same layout as 4.1exp
│
└── 4.5exp/                       # §4.5  Parallel DDM on CP³ (partition of unity)
    ├── HdDDM4.py                 #   Main program (parallel DDM, four subdomains)
    └── ...                       #   Same layout as 4.1exp
```

> The `error_curve/`, `un-u_inf_curve/`, `result/`, `models/`, `history_preds/` and `history_error/` subfolders are created/filled by the main scripts at run time.

## Requirements

The experiments were developed and run with the following environment:

| Package | Version |
|---|---|
| Python | 3.10.14 |
| torch | ≥ 2.0.0 (verified with `2.1.0.dev20230621+cu117` / CUDA 11.7 and `2.8.0+cu128` / CUDA 12.8) |
| numpy | 1.26.4 |
| matplotlib | 3.9.2 |

All Python dependencies are listed in [`requirements.txt`](requirements.txt). The code is GPU-accelerated with CUDA and falls back to CPU automatically, but a **CUDA GPU is strongly recommended** — the full experiments are computationally expensive (see [Notes](#notes)).

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv

#    Activate it:
#    - macOS / Linux:          source .venv/bin/activate
#    - Windows (cmd):          .venv\Scripts\activate
#    - Windows (PowerShell):   .venv\Scripts\Activate.ps1
#      (if PowerShell blocks scripts, run first:
#       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 2. Install the dependencies
pip install -r requirements.txt

# 3. (Optional) PyTorch build
#    `pip install -r requirements.txt` installs the default PyTorch from PyPI,
#    whose wheels bundle the CUDA runtime libraries — no separate CUDA Toolkit
#    installation is needed. An NVIDIA GPU driver must already be installed on
#    the system; check with `nvidia-smi`. Alternatives:
#    - CPU only:      pip install torch --index-url https://download.pytorch.org/whl/cpu
#    - CUDA 11.7:     pip install torch --index-url https://download.pytorch.org/whl/cu117
#      (note: the cu117 index only provides torch 2.0.1, the last build
#       that supports CUDA 11.7)
#    The code is verified with torch 2.1.0.dev20230621+cu117 and 2.8.0+cu128.
```

## Reproducing the Experiments

Each main script performs **5 independent trials** (random initialization) and generates all figures, logs and statistics automatically. Run each experiment from inside its own folder:

### Commands

| Paper section | Manifold | Folder | Command |
|---|---|---|---|
| §4.1 | S⁵ | `4.1exp` | `cd 4.1exp && python HdDDM1.py` |
| §4.2 | S¹⁰ | `4.2exp` | `cd 4.2exp && python HdDDM1.py` |
| §4.3 | B³ × S³ | `4.3exp` | `cd 4.3exp && python HdDDM2.py` |
| §4.4 | CP³ | `4.4exp` | `cd 4.4exp && python HdDDM3.py` |
| §4.5 | CP³ | `4.5exp` | `cd 4.5exp && python HdDDM4.py` |

Example:

```bash
cd 4.1exp
python HdDDM1.py
```

### Experiment Configuration

The default hyper-parameters hard-coded in each main script are summarized below (they correspond to the settings reported in the paper):

| § | Folder | Manifold | b | R | Network | points | interior points rate | interior loss weight (→ λ_BC = 1 − weight) | overlap s |
|---|---|---|---|---|---|---|---|---|---|
| 4.1 | `4.1exp` | S⁵ | 1.0 | 1.2 | FC-PINN, 500 × 4 | 2000 | 0.6 | 0.2 (→ 0.8) | — |
| 4.2 | `4.2exp` | S¹⁰ | 1.0 | 1.2 | FC-PINN, 500 × 6 | 5000 | 0.9 | 0.1 (→ 0.9) | — |
| 4.3 | `4.3exp` | B³ × S³ | 0.0 | 1.2 | Res-PINN, 500 × 2 | 2000 | 0.1 | 0.005 (→ 0.995) | — |
| 4.4 | `4.4exp` | CP³ | 4.0 | 1.2 | Res-PINN, 500 × 2 | 2000 | 0.5 | 0.003 (→ 0.997) | 1 |
| 4.5 | `4.5exp` | CP³ | 4.0 | 1.2 | Res-PINN, 500 × 2 | 2000 | 0.5 | 0.003 (→ 0.997) | 0.1 |

Common settings for all experiments:

- Loss: `L = interior_weight · L_PDE + (1 − interior_weight) · L_BC` — the paper's λ_BC is the weight of the boundary term, i.e. λ_BC = 1 − interior_weight (see eq. (3.4) of the paper).
- Inner iterations: 5000 epochs per outer step with the Adam optimizer, lr = 1e-3, exponential LR decay (γ = 0.9999), lr × 0.9 at each outer step.
- Activation: tanh. Sampling: uniform random points in balls / on spheres / in hypercubes (Monte Carlo).
- Outer iterations: 100 steps. Trials: 5 independent runs.

Problem settings (exact solutions and source terms):

| Manifold | Exact solution u | Source f |
|---|---|---|
| S⁵ / S¹⁰ | y<sub>d+1</sub> | (d+b)u |
| B³ × S³ | sin(π yₚ) + y'<sub>q+1</sub> | (b + π²)sin(π yₚ) + (b + q)y'<sub>q+1</sub> |
| CP³ | Σᵢ aᵢ\|wᵢ\|² | (4n + 4 + b)u − 4Σⱼ aⱼ, where a = (1, 2, −1, −2) |

**For the concrete settings of each experiment on its subdomains, see the paper and the file `HdDDM<n>.py` in the corresponding folder.**

### Outputs and Results

Each run of a main script produces the following files inside its experiment folder:

| Output | Location | Description |
|---|---|---|
| Relative L² error curves | `error_curve/` | `loss&error_curve_improved_all_<n>.png` (each subdomain), `..._max_<n>.png` (max over subdomains), `...mean_all_<st>-<ed>.png` (mean over the 5 trials for each subdomain), `..._mean_max_<st>-<ed>.png` (max of those means over the subdomains) |
| ‖uⁿ − u^∞‖ curves | `un-u_inf_curve/` | `un-u_inf__relative_l2_Over_Step_all_<n>.png`, `..._max_<n>.png`, plus mean versions |
| Per-trial log | `result/hdddm*_test_results.csv` | parameters, timestamp, wall-clock time and relative L² error of each trial |
| Statistics | `result/output.csv` | mean ± standard deviation of the max relative L² error over the subdomains at outer steps 0, 5, 10, 15, 20 and 100 (step 0 = the error of the random initialization) |
| Model checkpoints | `models/` | `models_<n>.pth` (state dicts of all subdomain networks) |
| Saved predictions | `history_preds/` | `historys_<n>.pth` (predictions at test points for each outer step) |
| Error histories | `history_error/` | `d_er_<n>.pkl` / `di_er_<n>.pkl` (per-step error histories, consumed by `data_processing.py`) |

## Notes

- **Compute cost.** Each trial runs 100 outer steps × 5000 inner epochs × (2–4) subdomain networks, repeated 5 times. A single full experiment can take many hours on a single GPU. The results reported in the paper were produced on a GPU cluster.
- **`run_count.txt`.** The main scripts keep a run counter in `run_count.txt` in the current working directory — always run them from inside the experiment folder (`cd 4.1exp` etc.) so that the counter and the output files stay in the right place.
- **Network architectures.** §4.1 and §4.2 use the plain fully-connected PINN, while §4.3, §4.4 and §4.5 use the residual network variant (`Res_PINN`, a ResNet-style network with shortcut connections). Both architectures are selectable via the `md` option (`'PINN'` / `'Res'`).

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{Jiang2026PDE,
  author        = {Jiang, Yufang and Qin, Lizhen and Wang, Feng},
  title         = {{Domain Decomposition Methods with Physics-Informed Neural Networks for Elliptic Equations on Manifolds}},
  journal       = {arXiv preprint},
  year          = {2026},
  eprint        = {2607.04285},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2607.04285}
}
```

The preprint PDF is included in this repository as [`DDM_PINN_arXiv.pdf`](DDM_PINN_arXiv.pdf).

## License

This project is released under the [MIT License](LICENSE).
