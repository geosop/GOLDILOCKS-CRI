# CRI-HYPOTHESIS

**Reproducibility pipeline for the "Conscious Retroactive Intervention (CRI)" Perspective / hypothesis manuscript in preparation**

---

## What this repository is

This repository hosts the full simulation and figure-generation stack for the CRI hypothesis:

> **“Conscious Retroactive Intervention: A Testable, Time-Symmetric Open Quantum Systems Framework for Predictive Coding.”**

The code implements the **Goldilocks-gated CRI model** on synthetic data only, providing:

- GKSL-style dynamics with a future-indexed dissipator and retro-horizon;
- “Goldilocks” logistic gating over anticipated futures;
- Tier-A (seconds-scale, sleep/quiet) and Tier-B (waking) synthetic EEG pipelines;  
- complete, scriptable reproduction of the main CRI figures.

This repository is meant as the **reproducibility and audit companion** to the CRI Perspective / hypothesis manuscript, and is written for readers in **cognitive neuroscience, psychology, and open quantum systems theory**.

> ⚠️ **Scope**  
> - Only _synthetic_ EEG and simulated data are used.  
> - No human or empirical datasets are shipped or analysed here.  
> - The code supports a theoretical hypothesis and its falsifiable predictions; it does **not** claim empirical confirmation.

Repository: <https://github.com/geosop/CRI-HYPOTHESIS>  

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/geosop/CRI-HYPOTHESIS.git
cd CRI-HYPOTHESIS

```
### 2. Create and activate the Conda environment
All dependencies are specified in `utilities/env-ci.yml`.

```bash
# One-time environment creation
conda env create -n cri_hypothesis -f utilities/env-ci.yml

# Activate the environment
conda activate cri_hypothesis
```
### 3. Run the full pipeline (one command)
This executes the end-to-end synthetic pipeline (Tier-A synthetic EEG, preprocessing, CRI simulations, statistics, and figures):

```bash
chmod +x run_all.sh
bash run_all.sh
```
Outputs:
- Figures (PDF/PNG): `figures/output/`
- Intermediate results and logs:`*/output/` subfolders in each module (e.g. `decay/output/`, `logistic_gate/output/`, `synthetic_EEG/output/`).

## Regenerating only the main figures
If you only need the core figures (and not every intermediate analysis), use the convenience script:

```bash
python generate_figures.py
```
This script re-runs the key figure drivers in `figures/` against already-generated synthetic data (or regenerates minimal inputs as needed).

## Repository Structure

```text
CRI-HYPOTHESIS/
├── .github/workflows/      # GitHub Actions CI (runs the pipeline on push/PR)
├── decay/                  # Retrocausal decay simulations & τ_fut fits
├── epochs_features/        # Epoch extraction & feature matrix construction
├── figures/                # All figure-generation scripts & outputs
├── logistic_gate/          # Goldilocks logistic gate simulations & fits
├── preprocessing/          # EEG preprocessing & artifact handling
├── qpt/                    # Quantum process tomography–style analyses
├── simulation_core/        # Core CRI dynamics and kernel weighting
├── statistics/             # Permutation tests, power analyses, summary stats
├── stimulus_presentation/  # PsychoPy cue-schedule generation (Tier A/B)
├── synthetic_EEG/          # Synthetic EEG generators (Tier A/B scenarios)
├── tierB_tempered/         # Tier-B waking / tempered-task simulations
├── utilities/              # Shared utilities, config, and environment files
├── .gitignore
├── LICENSE                 # MIT License
├── README.md               # This file
├── default_params.yml      # Master parameter file for all modules
├── generate_figures.py     # Convenience figure-generation entry point
├── run_all.sh              # Master pipeline script
└── smoke_test.log, local_smoke_test.log  # Example local test logs
```
## Module overview

- **`simulation_core/`**
  - CRI’s GKSL-style evolution, retro-horizon weighting, and Goldilocks gate inputs.
  - Implements the memory + sensory + future mixture evolution and computes the effective future weighting used for downstream analyses.

- **`decay/`**
  - Gate-on decay curves and log-linear fits to recover the effective future time constant \(\tau_{\mathrm{fut}}\) from synthetic data.
  - Supports Tier-A “seconds-scale” decay signatures.

- **`logistic_gate/`**
  - Goldilocks logistic gate simulations and fits vs. arousal / input probability.
  - Explores “too weak / just right / overloaded” regimes consistent with the Goldilocks gating concept.

- **`tierB_tempered/`**
  - Tier-B (waking, seconds-scale) CRI simulations for tempered task variants and seconds-scale gating.
  - Explores waking / task-based implementations of CRI-style retrocausal weighting.

- **`synthetic_EEG/`**
  - Generation of realistic synthetic EEG with configurable artifacts, noise, and spindles.
  - Designed for testing Tier-A/B pipelines without using empirical data.

- **`preprocessing/`**
  - MNE-style preprocessing, artifact detection, and bad-channel handling for synthetic EEG.
  - Supports robust bad-channel detection and expected warnings for synthetic data.

- **`epochs_features/`**
  - Epoching, feature matrices \(x_t\), and higher-level feature engineering required by statistics modules.
  - Bridges preprocessing outputs to CRI-specific analyses.

- **`qpt/`**
  - Surrogate quantum process tomography analyses over CRI’s future-indexed channel.
  - Matches the manuscript’s QPT-style signatures and open-systems diagnostics.

- **`statistics/`**
  - Permutation tests, power analyses, and summary statistics corresponding to CRI signatures (decay, gating, QPT).
  - Provides quantitative checks on hypothesis-level predictions.

- **`figures/`**
  - Drivers for all main and supplementary figures:
    - Decay curves.
    - Logistic gating surfaces.
    - Tier-A seconds-scale decay.
    - QPT schematics.
    - EEG preprocessing flowcharts.
  - Outputs are written into `figures/output/`.

- **`stimulus_presentation/`**
  - PsychoPy cue-schedule generator for Tier-A/B paradigms.
  - Generates synthetic cue schedules only (no human data).

- **`utilities/`**
  - Shared helper functions, logging, configuration utilities, and Conda environment specs (e.g. `env-ci.yml`).

- **`default_params.yml`**
  - Centralised parameter file (time constants, noise levels, thresholds, etc.) used across modules for consistent simulations.

## Running individual components

All commands below assume:

```bash
conda activate cri_hypothesis
```
## Synthetic EEG and preprocessing

```bash
# 1. Synthetic EEG generation
python synthetic_EEG/make_synthetic_eeg.py
python synthetic_EEG/visualize_synthetic_eeg.py

# 2. Preprocessing and artifact handling
python preprocessing/artifact_pipeline.py

# 3. Epoching and feature extraction
python epochs_features/extract_epochs.py
python epochs_features/compute_x_t.py
```
The synthetic pipelines include:
- Bad-channel detection using peak-to-peak ranges, flatness, and variance thresholds.
- Configurable thresholds in `default_params.yml`.
- Handling of bad channels via NaN assignments in the outputs rather than interpolation.

## Core CRI simulations

```bash
# Master equation and kernel weighting
python simulation_core/toy_model_master_eq.py
python simulation_core/retro_kernel_weight.py
```
### These scripts:
- Implement the memory + sensory + future mixture evolution.
- Compute effective future weights used to drive the Goldilocks gate and downstream analyses.

## Decay and logistic gating

```bash
# Gate-on decay and \(\tau_{\mathrm{fut}}\) fits
python decay/fit_decay.py
python decay/simulate_decay.py
python decay/wls_tobit_robustness.py

# Goldilocks logistic gate fits
python logistic_gate/fit_logistic.py
python logistic_gate/simulate_logistic.py
```
### These scripts:
- **Reproduce the Tier-A gate-on decay curves** and estimates of \(\tau_{\mathrm{fut}}\).
- Generate **Goldilocks logistic gating signatures** that the manuscript proposes as falsifiable predictions (seconds-scale Tier-A horizon, “just-right” regime, etc.).

## Quantum process tomography surrogate
```bash
python qpt/qpt_fit.py
python qpt/qpt_simulation.py
```
### These scripts:
- **Runs QPT-style surrogate identification** for the future-indexed channel.
- Supports open systems / Kossakowski-style analyses consistent with the manuscript and Supplementary Information.

## Tier-B / tempered-task simulations
```bash
python tierB_tempered/simulate_and_fit.py
```
### This module:
- Explores waking, seconds-scale tasks with tempered retrocausal weighting and logistic gating.
- Provides Tier-B predictions (see in-module docstrings for specific entry points and options).

## Main manuscript Figure 1 (TikZ → PDF & PNG)
A **TikZ program** has been added at `figures/CRI-manuscript_figure_1.tex` that compiles to a **vector PDF** and a **high-resolution PNG** into `figures/output/`.

### Continuous integration and downloadable artifacts
GitHub Actions workflows in `.github/workflows/` are configured to:

- Create a minimal CI environment from `utilities/env-ci.yml`.
- Run `run_all.sh` on every push or pull request to `main`.
- Upload the generated outputs (figures and key intermediates) as a single CI artifact.

### Artifact contents
- Figures are written to `figures/output/` as PDF/PNG files.
- CI bundles them into a zip archive named:

```text
pipeline-outputs-YYYY-MM-DD_HH-MM-SS
```


## One-click CI pipeline from GitHub Actions

You can run the full CRI pipeline directly on GitHub, without setting up a local environment, using the **“Run CRI pipeline (CI-safe)”** workflow:

1. Open the repository on GitHub and click the **Actions** tab.
2. In the left-hand sidebar, select **Run CRI pipeline (CI-safe)**.
3. At the top of the page, ensure **“Use workflow from”** is set to the `main` branch (default).
4. Click the green **Run workflow** button.

GitHub Actions will start a new run of the workflow defined in `.github/workflows/run_all.yml`, which in turn calls:
```bash
bash run_all.sh
```
on a fresh GitHub-hosted runner and produces updated figures and outputs as workflow artifacts. After the run completes, you can download the generated figures and other outputs from the Artifacts section of that workflow run (see “Download via the GitHub web UI” and “Download via GitHub CLI” below).

## Download via the GitHub web UI:
- Open the “Actions” tab of the repository.
- Click the most recent CI run on the `main` branch.
- Scroll to Artifacts and download the `pipeline-outputs-YYYY-MM-DD_HH-MM-SS` zip.
- Unzip and navigate to `figures/output/` to find the generated figures.

## Download via GitHub CLI (optional)
If you use the GitHub CLI (`gh`):
```bash
# Download the most recent run’s artifacts
gh run download

# Or download a specific artifact by name
gh run download -n "pipeline-outputs-YYYY-MM-DD_HH-MM-SS" -D ./artifacts
```
## Reproducibility, limitations, and intended use

- **Reproducible from code** – All figures and numerical analyses in the CRI Perspective / hypothesis manuscript are reproducible from this repository, using only synthetic data.
- **Open systems, not exotic physics** – The CRI model is implemented as a conventional open quantum systems description (GKSL-style generators, non-negative mixtures), without assuming long-lived macroscopic coherence.
- **Hypothesis-level status** – The code operationalises a set of testable predictions (e.g., decay constants, logistic gating regimes, QPT signatures) but does **not** provide empirical EEG or behavioural evidence.
- **No clinical / commercial validation** – Results are for research and hypothesis-generation only. Any clinical, applied, or commercial use would require independent validation and regulatory assessment.


All simulation code and figure-generation scripts are publicly available at
https://github.com/geosop/CRI-HYPOTHESIS
 (MIT License).
The repository includes a continuous-integration workflow (“Run CRI pipeline (CI-safe)”) that executes the full pipeline from a clean environment and regenerates all main and supplementary figures as downloadable artifacts for any given commit.


## Data and code availability

- **Code** – Publicly available under the MIT License at:  
  <https://github.com/geosop/CRI-HYPOTHESIS>
- **Data** – All data used by the scripts are **synthetic or simulated** and generated locally by the pipeline.
- **Figure-generation scripts** – All main figures can be reconstructed from the scripts in `figures/` plus the synthetic pipelines described above.

---
## EEG Artifact Handling Details
- Robust bad-channel detection for synthetic datasets.
- Peak-to-peak, flatness, and variance thresholds configurable in `default_params.yml`.
- In all simulation outputs, interpolation for bad channels is handled by assigning `NaN` values.

## Known Warnings & Expected Behavior
- `"Interpolation failed: No digitization points found..."` — Expected for synthetic datasets.
- Filename warnings from MNE — Outputs can be renamed for BIDS/MNE compatibility if required.
- `tight_layout` UserWarning — Figure rendering only; no effect on scientific results.
---

## Contact

**Maintained by:** George Sopasakis  
Conscious Retroactive Intervention Project, 2025

- For questions regarding possible collaborations, technical or reproducibility issues, please use the repository’s GitHub Issues page.


## How to cite

If you use this repository in academic work, please cite:

> Sopasakis, G., & Sopasakis, A. (2025). *Conscious Retroactive Intervention (CRI): a Testable, Time-Symmetric Open Quantum Systems Framework for Predictive Coding.* Perspective / hypothesis manuscript, in preparation. 
> GitHub: https://github.com/geosop/CRI-HYPOTHESIS


## Licensing

- **Code:** MIT license (see `LICENSE`).
- **Synthetic datasets:** CC BY 4.0 (see `DATA_LICENSE`). This applies to
  tabular data files (e.g. `.csv`, `.tsv`, `.txt`) and similar dataset
  formats throughout the repository, unless explicitly stated otherwise.

## Disclaimer

This repository contains research code for a Perspective manuscript under review. All data are synthetic or simulated; results are for demonstration and reproducibility purposes only. For clinical or commercial use, independent validation is required.

---
- **Previous Update:** 2025-07-24.
- **Last Update:** 2025-11-16.
