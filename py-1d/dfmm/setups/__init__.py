"""Initial-condition builders for the validation experiments.

Each module provides an IC constructor returning `(U, x)` where `U` has
shape `(8, N)` in the production eight-field convention and `x` is an
array of cell centers on the unit interval.

    sod        Sod shock tube, classical Riemann problem.
    sine       Cold sinusoidal velocity, shell-crossing / shock-forming test.
    shock      Steady-state shock with inflow-outflow boundaries.
    wavepool   Broadband random-phase Kolmogorov-like forcing, used for
               the Kramers-Moyal LES calibration and validation.
"""

from . import sod, sine, shock, wavepool

__all__ = ["sod", "sine", "shock", "wavepool"]
