"""
Per-cell closure diagnostics.

Given a state vector, `extract_diagnostics` returns the local quantities
needed to assess closure quality:

    Mach            |u| / c_s
    s               standardized skewness Q / (rho * Sigma_vv^{3/2}),
                    whose magnitude gauges the quartic max-entropy
                    realizability boundary (|s| < sqrt(2) approximately)
    gamma           Cholesky phase-space rank-loss diagnostic
    dudx            local velocity gradient (centered finite difference)
    rho, u, Pxx, Pp, Q
                    pass-through primitives for convenience

These are the scalars the paper uses as switching criteria for a
hybrid kinetic-fluid scheme and as predictors in the Kramers-Moyal
LES closure calibration.
"""
import numpy as np

from .schemes.cholesky import CSCOEF


def extract_diagnostics(U):
    """Compute all local closure diagnostics for a state vector.

    Parameters
    ----------
    U : ndarray, shape (8, N)

    Returns
    -------
    d : dict of ndarray
        Keys: rho, u, Pxx, Pp, Q, Mach, s, gamma, dudx
    """
    rho = U[0]
    u = U[1]/rho
    Pxx = U[2] - rho*u*u
    Pp = U[3]
    Q = U[7] - rho*u**3 - 3*u*Pxx
    cs = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))
    Svv = Pxx/np.maximum(rho, 1e-30)
    Svv_safe = np.maximum(Svv, 1e-30)
    beta = U[6]/rho
    gamma = np.sqrt(np.maximum(Svv - beta**2, 0))
    g_norm = gamma/np.sqrt(Svv_safe)
    s_vals = Q/(rho*Svv_safe**1.5)
    Mach = np.abs(u)/cs
    N = len(rho)
    dudx = np.gradient(u, 1.0/N)
    return dict(rho=rho, u=u, Pxx=Pxx, Pp=Pp, Q=Q,
                Mach=Mach, s=s_vals, gamma=g_norm, dudx=dudx)
