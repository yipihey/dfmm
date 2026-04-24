# momentlag

A one-dimensional moment scheme with built-in closure-quality diagnostics,
a Lagrangian-coordinate tracking of phase-space structure, and a
calibrated Kramers-Moyal large-eddy-simulation (LES) closure with
self-calibrating noise.

This repository contains everything needed to reproduce the paper
*"A Lagrangian-coordinate-aware moment scheme with built-in closure
diagnostics for moderate-Knudsen flows"*: the library code that
implements the scheme, numbered experiment scripts that produce every
figure, the LaTeX paper source, and minimal tests.

## What is inside

```
momentlag/
├── README.md                   This file.
├── LICENSE                     MIT.
├── Makefile                    Top-level build targets.
├── pyproject.toml              Python package metadata and dependencies.
├── momentlag/                  The Python library (the scheme).
│   ├── schemes/                Numerical schemes.
│   ├── setups/                 Initial-condition builders.
│   └── closure/                Kramers-Moyal LES closure machinery.
├── experiments/                Numbered reproduction scripts.
├── data/                       Small tracked calibration artifacts.
├── tests/                      Smoke tests (conservation, realizability).
├── paper/                      LaTeX source and figures.
└── docs/                       Additional reproduction notes.
```

## Quick start

```bash
# install the library in editable mode
pip install -e .

# run smoke tests (~10 s)
pytest tests/

# compile the paper PDF
make paper

# reproduce a single figure
python experiments/01_sod_validation.py
```

## The scheme in one sentence

We evolve the standard five conserved quantities of single-fluid
hydrodynamics, plus three phase-space diagnostics (a Lagrangian label
and its two Cholesky co-factors), plus the third velocity moment (the
heat flux). The Cholesky form makes realizability manifest; the
diagnostics fire only where the moment closure is breaking down, giving
the scheme a built-in self-report of its own validity.

## The LES closure in one sentence

By measuring the closure error in coarse-to-fine comparisons of a
wave-pool flow, we fit a calibrated stochastic correction with a drift
term $C_A\, \rho\, \partial_x u\, \Delta t$ and a compression-activated
noise term $C_B\, \rho\, \sqrt{\max(-\partial_x u,0)\, \Delta t}\, \eta$
with $\eta$ Laplace-distributed; the resulting noise-augmented coarse
scheme reproduces fine-DNS spectra up to factor $\sim 1.5$ across the
resolved range where the deterministic coarse scheme falls short by
two or more decades.

## Reproducing the paper

The full paper reproduction pipeline runs in about 20 minutes on a
single laptop core. See `experiments/README.md` for per-experiment
runtimes and dependencies. In short:

```bash
cd experiments
bash run_all.sh        # runs all experiments in order
cd ../paper
make                   # compiles paper.pdf from regenerated figures
```

Every figure in the paper is produced by exactly one script in
`experiments/`, numbered to match the paper's section order.

## Runtime environment

The code has been tested on Python 3.10–3.12. Dependencies are kept
minimal: NumPy, Numba, SciPy, and Matplotlib.
The scheme is written as a single Numba-JIT hot loop; first run incurs
compile overhead of a few seconds per kernel, subsequent runs are fast.
No MPI, no GPU, no external solver dependencies.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or the LES closure protocol, please cite the
accompanying paper (in preparation).

## Acknowledgements

Developed at KIPAC / Stanford University. The Lagrangian-coordinate
tracking idea builds on earlier joint work with J. Koh.
