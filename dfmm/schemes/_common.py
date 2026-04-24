"""Shared constants for the 8-field single-fluid moment state.

Imported by the cholesky scheme (`dfmm.schemes.cholesky`), the
energy-conservative noise scheme (`dfmm.closure.noise_model`), and
the two-fluid scheme (`dfmm.schemes.two_fluid`, where each species
occupies its own 8-field block). The maxent and barotropic schemes
use different wave-speed coefficients and do not share this module.

The IDX_* constants are plain ints; Numba inlines them into the
compiled code, so using `U[IDX_RHO]` has identical cost to `U[0]`.
"""
import numpy as np

IDX_RHO   = 0   # rho
IDX_MOM   = 1   # rho u
IDX_EXX   = 2   # E_xx = rho u^2 + P_xx
IDX_PP    = 3   # P_perp
IDX_L1    = 4   # rho L_1 (advected Lagrangian label)
IDX_ALPHA = 5   # rho alpha (Cholesky 11-component)
IDX_BETA  = 6   # rho beta  (Cholesky 21-component)
IDX_M3    = 7   # rho <v^3>

# 13-moment max-eigenvalue coefficient: cs = sqrt(CSCOEF * P_xx / rho)
CSCOEF = 3.0 + np.sqrt(6.0)
