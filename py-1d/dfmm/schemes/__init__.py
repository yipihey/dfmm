"""Numerical schemes for the dual-frame moment system.

Each module exposes a step function (the HLL+BGK update for one timestep)
and a convenience `run_to` driver that integrates to a target time and
saves snapshots.

Available schemes:

    cholesky       8-field production scheme with Cholesky-factored
                   phase-space covariance. Default for most applications.
    maxent         Same as cholesky but with polynomial-exponent
                   maximum-entropy fourth-moment closure.
    barotropic     5-field reduced scheme for barotropic equations of
                   state; first-moment closure is exact.
    two_fluid      Two-species extension with application-specific
                   cross-coupling kernels (dust, plasma, hard-sphere, SIDM).
"""

from . import cholesky, maxent, barotropic, two_fluid

__all__ = ["cholesky", "maxent", "barotropic", "two_fluid"]
