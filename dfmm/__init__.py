"""
dfmm (dual frame moment method): a Lagrangian-coordinate-aware moment
scheme with built-in closure-quality diagnostics and a calibrated
Kramers-Moyal large-eddy closure.

Top-level conventions
---------------------
State vector (8 fields, cell-centered, periodic or inflow/outflow):

    U[0] = rho                  (density)
    U[1] = rho u                (momentum)
    U[2] = E_xx = rho u^2 + P_xx  (longitudinal energy per volume)
    U[3] = P_perp               (transverse pressure per volume)
    U[4] = rho L_1              (advected Lagrangian label)
    U[5] = rho alpha            (Cholesky 11-component of Sigma)
    U[6] = rho beta             (Cholesky 21-component of Sigma)
    U[7] = M_3 = rho <v^3>      (third velocity moment)

Package layout
--------------
    dfmm.schemes           Numerical schemes (cholesky, maxent,
                           barotropic, two_fluid).
    dfmm.setups            IC builders for each test problem.
    dfmm.closure           Kramers-Moyal LES closure.
    dfmm.integrate         run_to, coarse_grain.
    dfmm.diagnostics       Closure-quality scalars (s, gamma, Mach).
    dfmm.analysis          Spectrum and Lagrangian-reindex helpers.
"""
__version__ = "0.1.0"

from . import schemes, setups, closure
from . import integrate, diagnostics, analysis

__all__ = [
    "schemes",
    "setups",
    "closure",
    "integrate",
    "diagnostics",
    "analysis",
]
