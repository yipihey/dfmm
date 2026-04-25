# Experiments

Numbered reproduction scripts for every figure in the paper. Each
script is self-contained; invoke any one from the repository root:

```bash
python experiments/03_steady_shock.py
```

Figures are written to `paper/figs/`, which the paper's `Makefile`
picks up when building the PDF.

## Mapping to paper figures

| Script | Paper section | Main figure produced | Runtime |
|--------|---------------|---------------------|---------|
| `01_sod_validation.py`       | §7.1 | `sod_three_tau.png`            | ~5 s |
| `02_sine_shell_crossing.py`  | §3   | `sine_three_tau.png`           | ~10 s |
| `03_steady_shock.py`         | §8.1 | `steady_shock_profiles.png`    | ~15 s |
| `04_shock_thickness_scans.py`| §8.5 | `shock_thickness_scans.png`    | ~5 min |
| `05_two_fluid_dust_gas.py`   | §10.5| `dust_gas_sinusoid.png`        | ~30 s |
| `06_maxent_kappa.py`         | §5   | `maxent_kappa.png`             | ~5 s |
| `07_eion_equilibration.py`   | §10.6| `eion_equilibration.png`       | ~30 s |
| `10_kmles_wavepool.py`       | §9.1 | `kmles_stageA_evolution.png`   | ~10 s |
| `11_kmles_calibrate.py`      | §9.5 | (stdout + diagnostic figure, no paper fig) | ~1 min |
| `12_kmles_correlation_scan.py`| §9.6| `kmles_stageF_corr.png`        | ~20 s |
| `13_kmles_energy_conservation.py` | §9.7 | `kmles_stageG_energy.png` | ~10 s |
| `14_kmles_ensemble.py`       | §9.8 | `kmles_stageH_ensemble.png`    | ~7 min (full) / ~10 s (SMALL=1) |

Some scripts produce additional figures (e.g., `03_steady_shock.py`
also produces `shock_diagnostics.png`, `shock_tau_scan.png`,
`convergence.png` for inspection). These are useful for digging into
the scheme behavior but are not referenced in the paper text.

## Running everything

```bash
bash run_all.sh
```

This runs experiments 01-14 sequentially; approximately 20 minutes on
a single laptop core. For a fast smoke test, set `SMALL=1`:

```bash
SMALL=1 bash run_all.sh
```

which reduces the ensemble size in experiment 14 (the other scripts are
already fast at default size).

## A note on experiment 11 and the Section 9 calibration

Experiment 11 (`11_kmles_calibrate.py`) reproduces the Kramers-Moyal
calibration protocol of Section 9.2-9.5 on a three-seed wave-pool
ensemble. It produces the three-panel calibration figure showing drift,
noise-amplitude, and residual distribution fits. The three-seed estimate
under-predicts the production coefficients `C_A, C_B` in
`data/noise_model_params.npz` by a factor of a few, because the
Lagrangian-frame density disambiguation requires ensemble averaging to
separate the drift from the chaotic-divergence floor. Experiments 12-14
use the production values stored in `data/noise_model_params.npz`.

## A note on experiment 14

Experiment 14 (`14_kmles_ensemble.py`) at full settings uses N_coarse=512
and 20 paired-phase realizations, taking ~7 minutes. The shipped
`kmles_stageH_ensemble.png` in `paper/figs/` was generated this way.
Running the experiment in SMALL mode (`SMALL=1`) reduces the
configuration to N_coarse=128 / 5 pairs and finishes in ~10 seconds but
produces noticeably noisier spectra; use SMALL=1 for smoke-testing, not
for the paper figure.
